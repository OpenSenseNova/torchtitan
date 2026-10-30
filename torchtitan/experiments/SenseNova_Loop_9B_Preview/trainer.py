# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""SFT/MTP training, fixed-cursor evaluation and metrics."""

import copy
import json
import logging
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.experiments.SenseNova_Loop_9B_Preview.engine import (
    LoopTrainer,
    LoopTrainingEngine,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.profiling import sample_step


class SFTTrainingEngine(LoopTrainingEngine):
    def prepare_step(self, global_valid_tokens, *, num_accumulation_steps=1):
        timer = getattr(self, "step_timing", None)
        with timer.phase("prepare") if timer is not None else nullcontext():
            return super().prepare_step(
                global_valid_tokens, num_accumulation_steps=num_accumulation_steps
            )

    def forward_backward_microbatch(self, **kwargs):
        timer = getattr(self, "step_timing", None)
        with timer.phase("microbatch") if timer is not None else nullcontext():
            return super().forward_backward_microbatch(**kwargs)

    def _non_pp_forward_backward_body(
        self, *, inputs, labels, model_kwargs, loss_kwargs
    ):
        timer = getattr(self, "step_timing", None)
        if timer is None or not timer.active:
            return super()._non_pp_forward_backward_body(
                inputs=inputs,
                labels=labels,
                model_kwargs=model_kwargs,
                loss_kwargs=loss_kwargs,
            )
        # Preserve Titan's train/typecheck contexts and exact loss/backward order.
        with self.train_context():
            with timer.phase("forward"):
                pred = self.model_parts[0](inputs, **model_kwargs)
            with timer.phase("loss"):
                loss, self.loss_metrics = self.loss_fn(pred, labels, **loss_kwargs)
            del pred
            with timer.phase("backward"), spmd.no_typecheck():
                loss.backward()
        return loss

    def save_checkpoint(self, *, last_step=False):
        start = time.perf_counter()
        saved = super().save_checkpoint(last_step=last_step)
        if saved and hasattr(self, "checkpoint_metrics_callback"):
            self.checkpoint_metrics_callback(time.perf_counter() - start)
        return saved

    def optimizer_step(self):
        timer = getattr(self, "step_timing", None)
        with timer.phase("optimizer") if timer is not None else nullcontext():
            return self._optimizer_step_with_diagnostics()

    def _optimizer_step_with_diagnostics(self):
        self.router_grad_norm = None
        if sample_step(
            self.num_completed_steps + 1, getattr(self, "diagnostics_interval", 1)
        ):
            squared = torch.zeros((), dtype=torch.float32, device=self.device)
            for model in self.model_parts:
                for name, param in model.named_parameters():
                    if ".mhc_" in name and param.grad is not None:
                        grad = (
                            param.grad.to_local()
                            if isinstance(param.grad, DTensor)
                            else param.grad
                        )
                        squared += grad.float().square().sum()
            dist.all_reduce(squared)
            # HSDP replicas own identical reduced gradients; count each shard once.
            squared /= self.parallel_dims.dp_replicate
            self.router_grad_norm = squared.sqrt().item()
        result = super().optimizer_step()
        self.peak_reserved = torch.cuda.max_memory_reserved()
        self.peak_allocated = torch.cuda.max_memory_allocated()
        return result


class SFTTrainer(LoopTrainer):
    engine_cls = SFTTrainingEngine

    def __init__(self, config, *, experiment_args):
        self.experiment_args = experiment_args
        self.validation_loader = None
        super().__init__(config)
        args = experiment_args
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.step_timing import (
            StepTiming,
        )

        self.step_timing = StepTiming(
            self.engine.device, enabled=getattr(args, "step_timing", False)
        )
        self.engine.step_timing = self.step_timing
        if self.step_timing.enabled:
            if (
                not config.training.disable_cuda_graphs
                or config.parallelism.pipeline_parallel_degree != 1
            ):
                raise ValueError("Step timing requires eager CUDA execution and PP=1")
            self.metrics_processor.step_timing = self.step_timing
            self.dataloader.step_timing = self.step_timing
            for model in getattr(self.engine, "model_parts", []):
                model.step_timing = self.step_timing
        self.engine.diagnostics_interval = args.diagnostics_interval
        self.engine.checkpoint_metrics_callback = self._record_checkpoint_metrics
        if dist.get_rank() == 0:
            log_dir = Path(args.output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix="experiment_metrics_", suffix=".jsonl", dir=log_dir, delete=False
            ) as stream:
                self.metrics_path = Path(stream.name)
            logging.getLogger(__name__).info(
                "Experiment metrics: %s", self.metrics_path
            )
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.profiling import (
            StepProfiler,
        )

        self.step_profiler = StepProfiler(
            directory=Path(args.output_dir) / "profiles",
            rank=dist.get_rank(),
            ranks=args.profile_ranks,
            start_step=args.profile_start_step,
            num_steps=args.profile_steps,
            device_type=self.engine.device.type,
            with_stack=args.profile_with_stack,
            profile_memory=args.profile_memory,
        )
        self._first_profile_step = True
        if args.val_dataset:
            if set(map(str, map(Path.resolve, map(Path, args.val_dataset)))) & set(
                map(str, map(Path.resolve, map(Path, args.cached_dataset)))
            ):
                raise ValueError(
                    "Validation and training dataset paths must be disjoint"
                )
            dims = self.engine.parallel_dims
            mesh = dims.get_mesh("dp") if dims.dp_enabled else None
            self.validation_loader = replace(
                config.dataloader, paths=args.val_dataset
            ).build(
                dp_world_size=mesh.size() if mesh is not None else 1,
                dp_rank=mesh.get_local_rank() if mesh is not None else 0,
                tokenizer=self.tokenizer,
                max_context_length=args.max_length,
                num_tokens_per_microbatch=args.max_length * args.micro_batch_size,
            )
            self.validation_start = copy.deepcopy(self.validation_loader.state_dict())

    @torch.no_grad()
    def evaluate_fixed(self):
        loader = self.validation_loader
        if loader is None:
            return
        engine = self.engine
        states = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state(),
        )
        mtp = engine.mtp_settings
        total = torch.zeros(4 if mtp else 2, dtype=torch.float64, device=engine.device)
        modes = [part.training for part in engine.model_parts]
        try:
            loader.load_state_dict(self.validation_start)
            for part in engine.model_parts:
                part.eval()
            for _ in range(self.experiment_args.eval_batches):
                batch = next(loader)
                inputs, labels, kwargs = engine.model_parts[0].preprocess_inputs(
                    batch.to_input_dict(engine.device),
                    parallel_dims=engine.parallel_dims,
                    parallelism=engine.config.parallelism,
                    max_num_documents=engine.max_num_documents,
                    max_context_length=self.experiment_args.max_length,
                    **engine.preprocess_inputs_kwargs,
                )
                with engine.train_context():
                    pred = engine.model_parts[0](inputs, **kwargs)
                    loss_kwargs = batch.to_loss_kwargs(engine.device)
                    if mtp:
                        # Collect raw sums; normalize once over the full evaluation.
                        one = torch.ones((), dtype=torch.int64, device=engine.device)
                        loss_kwargs.update(
                            global_valid_tokens=one, mtp_global_valid_tokens=one
                        )
                    loss, metrics = engine.loss_fn(pred, labels, **loss_kwargs)
                if mtp:
                    total += torch.stack(
                        [
                            metrics[key].double()
                            for key in (
                                "main_loss/sum",
                                "main_tokens/sum",
                                "mtp_loss/sum",
                                "mtp_tokens/sum",
                            )
                        ]
                    )
                else:
                    total[0] += loss.double()
                    total[1] += (labels != -100).sum()
            # TP=PP=1: labels and loss are each partitioned over CP and DP.
            dist.all_reduce(total)
            if total[1] <= 0:
                raise ValueError("Validation has no supervised tokens")
            value = (total[0] / total[1]).item()
            extra = {}
            if mtp:
                from torchtitan.experiments.SenseNova_Loop_9B_Preview.loss import (
                    normalize_mtp_losses,
                )

                objective, main, auxiliary = normalize_mtp_losses(
                    total[0],
                    total[2],
                    main_count=total[1],
                    mtp_count=total[3],
                    training_mode=mtp["training_mode"],
                    coefficient=mtp["loss_scaling_factor"],
                )
                value = float(objective)
                extra = {
                    "validation_metrics/loss_main": float(main),
                    "validation_metrics/loss_mtp": float(auxiliary),
                    "validation_metrics/loss_train": value,
                    "validation_metrics/main_valid_tokens": int(total[1]),
                    "validation_metrics/mtp_valid_tokens": int(total[3]),
                }
            if mtp:
                self.metrics_processor.log_validation(
                    loss=value, step=engine.num_completed_steps, extra_metrics=extra
                )
            else:
                self.metrics_processor.log_validation(
                    loss=value, step=engine.num_completed_steps
                )
            self.record(
                dict(
                    event="validation",
                    step=engine.num_completed_steps,
                    loss=value,
                    supervised_tokens=int(total[1].item()),
                    **extra,
                )
            )
        finally:
            for part, mode in zip(engine.model_parts, modes, strict=True):
                part.train(mode)
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])
            torch.cuda.set_rng_state(states[3])

    def record(self, values):
        if dist.get_rank() == 0:
            with self.metrics_path.open("a") as f:
                f.write(
                    json.dumps(
                        dict(time=datetime.now(timezone.utc).isoformat(), **values)
                    )
                    + "\n"
                )

    def _record_checkpoint_metrics(self, seconds):
        self.metrics_processor.record_checkpoint(seconds)
        duration = torch.tensor(seconds, dtype=torch.float64, device=self.engine.device)
        if dist.is_initialized():
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        self.record(
            dict(
                event="checkpoint",
                step=self.engine.num_completed_steps,
                checkpoint_seconds=duration.item(),
            )
        )
        logging.getLogger(__name__).info(
            "Checkpoint step %d: save %.3fs",
            self.engine.num_completed_steps,
            duration.item(),
        )

    def microbatch_generator(self, data_iterable):
        iterator = super().microbatch_generator(data_iterable)
        while True:
            with self.step_timing.phase("dataloader"):
                batch = next(iterator)
            self.step_timing.record_batch(batch.document_lengths, batch.input.numel())
            model = self.engine.model_parts[0]
            repeat = model.loop_settings.count_for_step(self.engine.num_completed_steps)
            self.metrics_processor.record_batch(
                batch.document_lengths, repeat, model.loop_flops
            )
            yield batch

    def train(self):
        super().train()
        if self.engine.num_completed_steps % self.experiment_args.eval_steps:
            self.evaluate_fixed()

    def on_checkpoint_loaded(self, resumed):
        super().on_checkpoint_loaded(resumed)
        self.evaluate_fixed()

    def train_step(self, data_iterator):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.mhc.model import (
            MHCRouter,
        )

        collect = sample_step(
            self.engine.num_completed_steps + 1,
            self.experiment_args.diagnostics_interval,
        )
        for part in self.engine.model_parts:
            for module in part.modules():
                if isinstance(module, MHCRouter):
                    module.collect_diagnostics = collect
        next_step = self.engine.num_completed_steps + 1
        if self._first_profile_step:
            if (
                self.step_profiler.start_step
                and self.step_profiler.start_step < next_step
            ):
                raise ValueError(
                    "Profiler start step must be after the resumed checkpoint step"
                )
            self._first_profile_step = False
        with self.metrics_processor.exclude_time():
            self.step_profiler.begin(next_step)
        self.metrics_processor.begin_step()
        self.step_timing.begin_step()
        try:
            super().train_step(data_iterator)
        finally:
            self.step_timing.end_step()
        step = self.engine.num_completed_steps
        values = dict(
            event="train",
            step=step,
            router_grad_norm=self.engine.router_grad_norm,
            learning_rates=self.engine.lr_schedulers.get_metrics(),
            profiled=self.step_profiler.contains(step),
        )
        # Values here are the next update's LRs; native Titan logs the used LRs.
        values["learning_rates_phase"] = "next_update"
        peaks = torch.tensor(
            [self.engine.peak_reserved, self.engine.peak_allocated],
            dtype=torch.float64,
            device=self.engine.device,
        )
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        values["max_reserved_gib"], values["max_allocated_gib"] = (
            peaks / 2**30
        ).tolist()
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.mhc.model import (
            MHCRouter,
        )

        diagnostics = []
        for part in self.engine.model_parts:
            for module in part.modules():
                if (
                    collect
                    and isinstance(module, MHCRouter)
                    and hasattr(module, "last_diagnostics")
                ):
                    diagnostics.append(module.last_diagnostics)
        if diagnostics:
            stats = torch.stack(diagnostics).mean(0)
            dist.all_reduce(stats)
            stats /= dist.get_world_size()
            values["routing"] = dict(
                zip(
                    ("row_sum_error", "column_sum_error", "stream_variance"),
                    stats.tolist(),
                )
            )
        if getattr(self.metrics_processor, "last_metrics_step", None) == step:
            metrics = self.metrics_processor.last_metrics
            values.update(
                seconds=metrics["time_metrics/end_to_end(s)"],
                interval_steps=metrics["time_metrics/interval_steps"],
                interval_seconds=metrics["time_metrics/interval(s)"],
                TGS=metrics["throughput(TGS)"],
                valid_TGS=metrics["throughput(valid_TGS)"],
                train_seconds=metrics["time_metrics/train(s)"],
                checkpoint_seconds=metrics["time_metrics/checkpoint(s)"],
                padding_percent=metrics["packing/padding(%)"],
                mfu=metrics.get("mfu(%)"),
                tflops=metrics["tflops"],
                metrics=metrics,
                mfu_scope="useful_decoder_loop_packed_v1",
            )
            if self.step_timing.enabled:
                values["rank_timing"] = self.step_timing.last_rank_values
            self.record(values)
        with self.metrics_processor.exclude_time():
            self.step_profiler.end(step)
        if step % self.experiment_args.eval_steps == 0:
            self.evaluate_fixed()

    def close(self):
        if hasattr(self, "step_profiler"):
            self.step_profiler.close()
        if self.validation_loader is not None:
            self.validation_loader.close()
        super().close()
