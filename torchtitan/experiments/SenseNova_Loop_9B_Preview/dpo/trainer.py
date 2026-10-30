# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Pair-step scheduling, evaluation and metrics for native Loop DPO."""

import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from torchtitan.experiments.SenseNova_Loop_9B_Preview.engine import LoopTrainer
from .engine import DPOTrainingEngine
from .metrics import add_metrics, reduce_metrics, synchronized_error
from .reference import preserve_rng

logger = logging.getLogger(__name__)


class DPOTrainer(LoopTrainer):
    engine_cls = DPOTrainingEngine

    def __init__(self, config, *, dpo_args):
        self.dpo_args = dpo_args
        super().__init__(config)
        self.val_loader = None
        if dpo_args.val_dataset:
            self.val_loader = replace(
                config.dataloader, paths=dpo_args.val_dataset
            ).build(
                dp_world_size=self.dataloader.dp_world_size,
                dp_rank=self.dataloader.dp_rank,
                tokenizer=self.tokenizer,
                max_context_length=config.training.max_context_length,
                num_tokens_per_microbatch=config.training.num_tokens_per_microbatch_per_dp_rank,
            )

    def on_checkpoint_loaded(self, resumed):
        self.engine.initialize_reference()
        if not self.config.checkpointer.load_only:
            self.engine.checkpointer.ensure_reference_snapshot()
        logger.info(
            "DPO ready: beta=%s, reference frozen, batches count preference pairs, resumed=%s",
            self.config.loss.beta,
            resumed,
        )
        self.evaluate()

    def _step_batches(self, loader):
        error = None
        try:
            batches, global_pairs = loader.next_step()
        except Exception as exception:
            error = exception
        synchronized_error(error, self.engine.device)
        count = torch.tensor(len(batches), device=self.engine.device)
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
        batches += [loader.collate([]) for _ in range(int(count.item()) - len(batches))]
        local = torch.tensor(
            [
                sum(batch.num_valid_pairs for batch in batches),
                sum(batch.num_valid_tokens for batch in batches),
            ],
            device=self.engine.device,
        )
        mesh = self.engine.parallel_dims.get_optional_mesh("dp")
        if mesh is not None:
            dist.all_reduce(local, group=mesh.get_group())
        if local[0].item() != global_pairs or global_pairs <= 0:
            raise ValueError("DPO step plan has inconsistent global pair membership")
        return batches, global_pairs, int(local[1].item())

    def _log(self, metrics, *, phase):
        record = dict(step=self.engine.num_completed_steps, phase=phase, **metrics)
        if dist.get_rank() == 0:
            with (Path(self.config.dump_folder) / "dpo_metrics.jsonl").open(
                "a"
            ) as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            logger.info(
                "%s DPO step=%d loss=%.7f margin=%.6f accuracy=%.4f pairs=%d",
                phase,
                record["step"],
                metrics["loss"],
                metrics["reward_margin"],
                metrics["preference_accuracy"],
                metrics["valid_pairs"],
            )
        self.metrics_processor.logger.log(
            {
                f"{phase}/{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
            self.engine.num_completed_steps,
        )

    def train_step(self, data_iterator):
        start = time.perf_counter()
        batches, global_pairs, global_tokens = self._step_batches(self.dataloader)
        engine = self.engine
        engine.prepare_dpo_step(
            global_pairs, global_tokens, num_accumulation_steps=len(batches)
        )
        loop_count = engine.model_parts[0].loop_train_count
        totals = {}
        for index, batch in enumerate(batches):
            add_metrics(
                totals,
                engine.preference_microbatch(
                    batch,
                    global_pairs=global_pairs,
                    loop_count=loop_count,
                    backward=True,
                    accumulation_index=index,
                ),
            )
        grad_norm = engine.optimizer_step()
        metrics = reduce_metrics(totals, engine.parallel_dims)
        elapsed = time.perf_counter() - start
        metrics.update(
            loop_count=loop_count,
            grad_norm=float(grad_norm),
            pairs_per_second=global_pairs / elapsed,
            completion_tokens_per_second=(
                metrics["chosen_tokens"] + metrics["rejected_tokens"]
            )
            / elapsed,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            pair_ids=[identity for batch in batches for identity in batch.pair_ids],
        )
        self._log(metrics, phase="train")
        if engine.num_completed_steps % self.dpo_args.eval_steps == 0:
            self.evaluate()

    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            return
        engine = self.engine
        state = self.val_loader.state_dict()
        modes = [model.training for model in engine.model_parts]
        totals = {}
        try:
            with preserve_rng():
                self.val_loader.epoch = self.val_loader.cursor = 0
                self.val_loader._set_order()
                for model in engine.model_parts:
                    model.eval()
                for _ in range(self.dpo_args.eval_batches):
                    batches, count, _ = self._step_batches(self.val_loader)
                    for batch in batches:
                        add_metrics(
                            totals,
                            engine.preference_microbatch(
                                batch,
                                global_pairs=count,
                                loop_count=engine.model_parts[
                                    0
                                ].loop_settings.eval_count,
                                backward=False,
                            ),
                        )
                    if self.val_loader.cursor == len(self.val_loader.dataset):
                        break
                metrics = reduce_metrics(totals, engine.parallel_dims)
                if not all(
                    torch.isfinite(torch.tensor(value)) for value in metrics.values()
                ):
                    raise ValueError("Nonfinite DPO evaluation scores")
                self._log(metrics, phase="eval")
        finally:
            self.val_loader.load_state_dict(state)
            for model, mode in zip(engine.model_parts, modes):
                model.train(mode)

    def close(self):
        if self.val_loader is not None:
            self.val_loader.close()
        super().close()
