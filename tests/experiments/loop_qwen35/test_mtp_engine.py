# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer-step lifecycle contracts shared by Loop MTP and the base trainer."""

import copy
import pytest
import torch
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Select the repository under test.

from torchtitan.trainer import Trainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


class OptimizerGroup(list):
    def step(self):
        for optimizer in self:
            optimizer.step()

    def zero_grad(self, **kwargs):
        for optimizer in self:
            optimizer.zero_grad(**kwargs)


def small_engine(mode="mtp_only"):
    from torchtitan.experiments.loop_qwen35.engine import LoopTrainingEngine
    from test_mtp_model import structural_model

    model = structural_model(mode)
    model.mtp_settings["loss_scaling_factor"] = 1.0 if mode == "mtp_only" else 0.1
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01, weight_decay=0.2
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1 / (1 + step)
    )
    engine = object.__new__(LoopTrainingEngine)
    engine.model_parts, engine.optimizers = [model], OptimizerGroup([optimizer])
    engine.lr_schedulers = scheduler
    engine.parallel_dims = SimpleNamespace(
        tp=1, pp_enabled=False, ep_enabled=False, get_optional_mesh=lambda name: None
    )
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        training=SimpleNamespace(max_norm=1.0, disable_cuda_graphs=True)
    )
    engine.gc_handler = SimpleNamespace(run=lambda step: None)
    engine.sdc_replayer = None
    engine.num_completed_steps = engine.ntokens_seen = (
        engine._num_optimizer_steps_since_cuda_graph_init
    ) = 0
    engine.optimizer_updates = engine.skipped_empty_mtp_steps = 0
    return engine, model, optimizer, scheduler


def assert_state_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for value, other in zip(actual, expected):
            assert_state_equal(value, other)
    else:
        assert actual == expected


def _tp_fsdp_norm_worker(rank, rendezvous):
    import spmd_types as spmd
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    from torchtitan.models.common.decoder_sharding import dense_param_placement
    from torchtitan.protocols.sharding import ShardingConfig

    from torchtitan.experiments.loop_qwen35.engine import (
        LoopTrainingEngine,
        synchronize_replicated_tp_gradients,
    )

    torch.set_num_threads(1)
    store = dist.TCPStore("127.0.0.1", rendezvous, is_master=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=4)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "tp"))
        dp_mesh, tp_mesh = mesh["dp_shard"], mesh["tp"]
        tp_rank = tp_mesh.get_local_rank()
        for fsdp in (False, True):
            holder = nn.Module()
            holder.mtp_layers = nn.ModuleList([nn.Module()])
            holder.frozen = nn.Linear(4, 4, bias=False).requires_grad_(False)
            holder.frozen.weight.grad = torch.full_like(holder.frozen.weight, 100.0)
            frozen_gradient = holder.frozen.weight.grad.clone()
            references = []
            local_expected = []
            for index, layout in enumerate((spmd.R, spmd.I, spmd.S(1))):
                complete = torch.arange(16).reshape(4, 4).float() / 100 + 0.1 * (
                    index + 1
                )
                references.append(complete)
                local = (
                    complete.chunk(2, dim=1)[tp_rank].contiguous()
                    if index == 2
                    else complete
                )
                gradient = local / 2 if index == 0 else local
                parameter_data = torch.ones_like(local)
                if fsdp:
                    parameter_data = distribute_tensor(
                        parameter_data, dp_mesh, [Shard(0)]
                    )
                    gradient = distribute_tensor(gradient, dp_mesh, [Shard(0)])
                module = nn.Module()
                module.weight = nn.Parameter(parameter_data)
                module.weight.grad = gradient.clone()
                module._sharding_config = ShardingConfig(
                    state_shardings={"weight": dense_param_placement(tp=layout)}
                )
                holder.mtp_layers[0].add_module(str(index), module)
                local_expected.append(local)
            # The same physical parameter registered twice must be reduced once.
            holder.alias = holder.mtp_layers[0].get_submodule("0")
            synchronize_replicated_tp_gradients([holder], tp_mesh)
            engine = object.__new__(LoopTrainingEngine)
            engine.model_parts = [holder]
            engine.device = torch.device("cpu")
            engine.config = SimpleNamespace(training=SimpleNamespace(max_norm=0.5))
            engine.parallel_dims = SimpleNamespace(tp=2, get_mesh=lambda name: tp_mesh)
            expected_norm = torch.cat([value.flatten() for value in references]).norm()
            norm = engine.clip_grad_norm()
            torch.testing.assert_close(norm, expected_norm, atol=1e-5, rtol=1e-4)
            coefficient = (0.5 / (expected_norm + 1e-6)).clamp(max=1)
            for module, expected in zip(
                holder.mtp_layers[0].children(), local_expected
            ):
                gradient = (
                    module.weight.grad.full_tensor() if fsdp else module.weight.grad
                )
                torch.testing.assert_close(
                    gradient, expected * coefficient, atol=1e-5, rtol=1e-4
                )
            assert torch.equal(holder.frozen.weight.grad, frozen_gradient)
    finally:
        dist.destroy_process_group()


def test_tp_partial_invariant_and_fsdp_norm():
    import torch.distributed as dist
    import torch.multiprocessing as mp

    store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    mp.spawn(_tp_fsdp_norm_worker, args=(store.port,), nprocs=4, join=True)


class EmptyMTPStepTest(unittest.TestCase):
    def test_empty_step_keeps_adam_momentum_weights_and_scheduler_exact(self):
        engine, model, optimizer, scheduler = small_engine()
        for empty in (False, True, False):
            before_step = engine.num_completed_steps
            engine.mtp_global_valid_tokens = torch.tensor(0 if empty else 2)
            engine.prepare_step(torch.tensor(3))
            engine.loss_is_finite = torch.ones((), dtype=torch.int32)
            tokens, positions = torch.tensor([1, 2, 3, 4]), torch.arange(4)
            _, hidden = model(
                tokens, positions=positions, mtp_input=tokens, mtp_positions=positions
            )
            labels = (
                torch.full((4,), -100) if empty else torch.tensor([3, 4, -100, -100])
            )
            loss = torch.nn.functional.cross_entropy(
                model.lm_head(hidden), labels, reduction="sum"
            )
            loss.backward()
            before = copy.deepcopy(
                (model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
            )
            if empty:
                assert optimizer.state and any(
                    torch.count_nonzero(state["exp_avg"])
                    for state in optimizer.state.values()
                )
            norm = engine.optimizer_step()
            assert (
                torch.isfinite(norm) and engine.num_completed_steps == before_step + 1
            )
            if empty:
                assert_state_equal(
                    (
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                    ),
                    before,
                )
                assert all(p.grad is None for p in model.parameters())
                assert (engine.optimizer_updates, engine.skipped_empty_mtp_steps) == (
                    1,
                    1,
                )
        assert (
            engine.num_completed_steps,
            engine.optimizer_updates,
            engine.skipped_empty_mtp_steps,
        ) == (3, 2, 1)
        saved = engine.state_dict()
        engine.optimizer_updates = engine.skipped_empty_mtp_steps = 0
        engine.load_state_dict(saved)
        assert engine.optimizer_updates == 2 and engine.skipped_empty_mtp_steps == 1

    def test_joint_empty_mtp_still_updates_main(self):
        engine, model, optimizer, scheduler = small_engine("joint")
        engine.mtp_global_valid_tokens = torch.tensor(0)
        engine.prepare_step(torch.tensor(2))
        engine.loss_is_finite = torch.ones((), dtype=torch.int32)
        tokens, positions = torch.tensor([1, 2, 3, 4]), torch.arange(4)
        hidden, auxiliary = model(
            tokens, positions=positions, mtp_input=tokens, mtp_positions=positions
        )
        loss = torch.nn.functional.cross_entropy(
            model.lm_head(hidden), torch.tensor([3, 4, -100, -100])
        )
        (loss + auxiliary.sum() * 0).backward()
        before = model.tok_embeddings.weight.detach().clone()
        engine.optimizer_step()
        assert not torch.equal(before, model.tok_embeddings.weight)
        assert (engine.optimizer_updates, engine.skipped_empty_mtp_steps) == (1, 0)

    def test_nonfinite_empty_step_still_rejected(self):
        engine, model, _, _ = small_engine()
        engine.mtp_global_valid_tokens = torch.tensor(0)
        engine.prepare_step(torch.tensor(1))
        engine.loss_is_finite = torch.zeros((), dtype=torch.int32)
        for parameter in model.mtp_layers.parameters():
            parameter.grad = torch.zeros_like(parameter)
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            engine.optimizer_step()
        assert engine.num_completed_steps == engine.skipped_empty_mtp_steps == 0


class MTPStepMetricsTest(unittest.TestCase):
    def test_accumulates_all_microbatches_and_logs_mode_equation(self):
        from torchtitan.training_engine import TrainingEngine

        from torchtitan.experiments.loop_qwen35.engine import LoopTrainer

        for mode in ("mtp_only", "joint"):
            engine, _, _, _ = small_engine(mode)
            engine.mtp_settings["loss_scaling_factor"] = 0.3
            engine.mtp_global_valid_tokens = torch.tensor(3)
            engine.prepare_step(torch.tensor(4))
            contributions = ((9.0, 4.0, 3, 1), (3.0, 12.0, 1, 2))

            def forward(**kwargs):
                values = contributions[kwargs["accumulation_index"]]
                engine.loss_metrics = dict(
                    zip(
                        (
                            "main_loss/sum",
                            "mtp_loss/sum",
                            "main_tokens/sum",
                            "mtp_tokens/sum",
                        ),
                        (torch.tensor(value) for value in values),
                    )
                )
                return torch.tensor(1.0)

            with patch.object(
                TrainingEngine, "forward_backward_microbatch", side_effect=forward
            ):
                engine.forward_backward_microbatch(accumulation_index=0)
                engine.forward_backward_microbatch(accumulation_index=1)
            report = engine.mtp_step_statistics()
            assert report["loss/main_sum"] == 12 and report["loss/mtp_sum"] == 16
            assert report["tokens/main_valid"] == 4 and report["tokens/mtp_valid"] == 3
            expected = 0.3 * 16 / 3 + (3 if mode == "joint" else 0)
            self.assertAlmostEqual(report["loss/train"], expected)
            trainer = object.__new__(LoopTrainer)
            engine.mtp_grad_norm = torch.tensor(2.0)
            trainer.engine = engine
            trainer.metrics_processor = SimpleNamespace(log=Mock())
            trainer.log_step_metrics(
                accumulated_loss=torch.tensor(expected),
                local_valid_tokens=4,
                global_valid_tokens=torch.tensor(4),
                grad_norm=torch.tensor(2.0),
                lr_metrics={"lr": 0.01},
            )
            assert engine.last_mtp_step_metrics["mtp/training_mode"] == mode
            assert trainer.metrics_processor.log.call_args.args[1] == expected
            assert all(
                isinstance(value, (int, float))
                for value in trainer.metrics_processor.log.call_args.kwargs[
                    "extra_metrics"
                ].values()
            )

    def test_empty_objective_is_zero_and_count_mismatch_rejected(self):
        engine, _, _, _ = small_engine()
        engine.main_global_valid_tokens = torch.tensor(4)
        engine.mtp_global_valid_tokens = torch.tensor(0)
        engine.mtp_step_metrics = dict(
            zip(
                ("main_loss/sum", "mtp_loss/sum", "main_tokens/sum", "mtp_tokens/sum"),
                (
                    torch.tensor(12.0),
                    torch.tensor(0.0),
                    torch.tensor(4),
                    torch.tensor(0),
                ),
            )
        )
        report = engine.mtp_step_statistics()
        assert (
            report["loss/main"] == 3
            and report["loss/train"] == report["loss/max_local_train"] == 0
        )
        engine.mtp_global_valid_tokens = torch.tensor(1)
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            engine.mtp_step_statistics()


@pytest.mark.parametrize("accumulation", [1, 2, 4])
@pytest.mark.parametrize("chunks", [0, 1, 4])
def test_real_trainer_consumes_empty_step_without_optimizer_or_scheduler_update(
    accumulation, chunks
):
    from torchtitan.components.data.types import TokenizedTrainingMicrobatch
    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.config import ParallelismConfig

    from torchtitan.experiments.loop_qwen35.engine import LoopTrainer
    from torchtitan.experiments.loop_qwen35.loss import LoopMTPLoss
    from torchtitan.experiments.loop_qwen35.mtp_data import align_mtp_batch

    torch.manual_seed(42)
    engine, model, optimizer, scheduler = small_engine()
    engine.parallel_dims.dp_enabled = False
    engine.parallel_dims.dp_replicate_enabled = False
    engine.parallel_dims.cp = 1
    engine.config.parallelism = ParallelismConfig()
    engine.config.training.max_context_length = 4
    engine.config.training.num_tokens_per_microbatch_per_dp_rank = 4
    engine.max_num_documents = None
    engine.preprocess_inputs_kwargs = {}
    engine.train_context = nullcontext
    engine.forward_backward_body_fn = engine._non_pp_forward_backward_body
    config = LoopMTPLoss.Config(
        training_mode="mtp_only", loss_scaling_factor=1.0, global_vocab_size=16
    )
    engine.loss_fn = config.build()
    model._skip_lm_head = bool(chunks)
    if chunks:
        engine.loss_fn = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(num_chunks=chunks, loss_fn=config)
        )
        engine.loss_fn.set_lm_head(model.lm_head)

    def preprocess(batch, **kwargs):
        aligned = align_mtp_batch(batch)
        return (
            batch["input"],
            (batch["labels"], aligned["mtp_labels"]),
            {
                "positions": batch["positions"],
                "mtp_input": aligned["mtp_input"],
                "mtp_positions": aligned["mtp_positions"],
            },
        )

    model.preprocess_inputs = preprocess
    engine.lr_schedulers = SimpleNamespace(
        step=scheduler.step, get_metrics=lambda: {"lr": optimizer.param_groups[0]["lr"]}
    )
    trainer = object.__new__(LoopTrainer)
    trainer.engine = engine
    trainer.gradient_accumulation_steps, trainer.num_pp_microbatches = accumulation, 1
    trainer.metrics_processor = SimpleNamespace(
        should_log=lambda step: True, log=Mock()
    )

    class Batches:
        cursor = 0

        def __iter__(self):
            return self

        def __next__(self):
            empty = self.cursor // accumulation == 1
            self.cursor += 1
            labels = (
                torch.tensor([3, -100, -100, -100])
                if empty
                else torch.tensor([-100, 3, 5, -100])
            )
            return TokenizedTrainingMicrobatch(
                input=torch.tensor([1, 3, 5, 7]),
                labels=labels,
                positions=torch.arange(4),
                padding_mask=torch.zeros(4, dtype=torch.bool),
                num_valid_tokens=int((labels != -100).sum()),
            )

    batches = Batches()
    for step in range(3):
        before = copy.deepcopy(
            (model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
        )
        trainer.train_step(batches)
        assert batches.cursor == (step + 1) * accumulation
        report = engine.last_mtp_step_metrics
        assert report["loss/train"] == report["loss/mtp"] == report["loss/total"]
        assert report["steps/consumed"] == step + 1
        if step == 1:
            assert optimizer.state
            assert_state_equal(
                (model.state_dict(), optimizer.state_dict(), scheduler.state_dict()),
                before,
            )
            assert (
                report["tokens/mtp_valid"]
                == report["loss/train"]
                == report["mtp/grad_norm"]
                == 0
            )
            assert (
                report["steps/optimizer_updates"]
                == report["steps/skipped_empty_mtp"]
                == 1
            )
    assert engine.ntokens_seen == 3 * accumulation * 4
    assert (engine.optimizer_updates, engine.skipped_empty_mtp_steps) == (2, 1)


class TrainerHookTest(unittest.TestCase):
    def make_trainer(self, *, log=True, hook=False):
        events = []

        class HookTrainer(Trainer):
            def prepare_step_microbatches(self, groups):
                events.append(
                    (
                        "prefetched",
                        [[batch.index for batch in group] for group in groups],
                    )
                )
                return super().prepare_step_microbatches(groups)

        trainer = object.__new__(HookTrainer if hook else Trainer)
        trainer.gradient_accumulation_steps = 2
        trainer.num_pp_microbatches = 2
        engine = SimpleNamespace(
            num_completed_steps=0,
            device="cpu",
            ntokens_seen=16,
            parallel_dims=SimpleNamespace(dp_enabled=False, dp_cp_enabled=False),
        )

        def prepare(count, *, num_accumulation_steps):
            events.append(("prepare", int(count), num_accumulation_steps))
            return count

        def forward(*, microbatch_group, global_valid_tokens, accumulation_index):
            events.append(
                (
                    "forward",
                    [batch.index for batch in microbatch_group],
                    int(global_valid_tokens),
                    accumulation_index,
                )
            )
            return torch.tensor(0.125)

        def update():
            events.append(("update",))
            engine.num_completed_steps += 1
            return torch.tensor(2.0)

        def learning_rates():
            events.append(("lr",))
            return {"lr": 0.01}

        engine.prepare_step = prepare
        engine.forward_backward_microbatch = forward
        engine.optimizer_step = update
        engine.lr_schedulers = SimpleNamespace(get_metrics=learning_rates)
        trainer.engine = engine
        trainer.metrics_processor = SimpleNamespace(
            should_log=lambda step: log, log=Mock()
        )

        def batches():
            for index in range(6):
                events.append(("fetch", index))
                yield SimpleNamespace(index=index, num_valid_tokens=index + 1)

        return trainer, events, batches()

    def test_default_prefetch_hook_is_identity(self):
        trainer, _, _ = self.make_trainer()
        groups = [[object()], [object()]]
        self.assertIs(trainer.prepare_step_microbatches(groups), groups)

    def test_consumption_update_and_default_metrics_unchanged(self):
        for hook in (False, True):
            for log in (False, True):
                with self.subTest(hook=hook, log=log):
                    trainer, events, iterator = self.make_trainer(hook=hook, log=log)
                    with patch(
                        "torchtitan.trainer.collect_aux_loss_metrics",
                        return_value={"aux": 3.0},
                    ):
                        trainer.train_step(iterator)
                    expected = [("fetch", i) for i in range(4)]
                    if hook:
                        expected += [("prefetched", [[0, 1], [2, 3]])]
                    expected += [
                        ("prepare", 10, 2),
                        ("forward", [0, 1], 10, 0),
                        ("forward", [2, 3], 10, 1),
                    ]
                    if log:
                        expected += [("lr",)]
                    expected += [("update",)]
                    self.assertEqual(events, expected)
                    self.assertEqual(trainer.engine.num_completed_steps, 1)
                    if log:
                        trainer.metrics_processor.log.assert_called_once_with(
                            1,
                            0.25,
                            0.25,
                            2.0,
                            extra_metrics={"n_tokens_seen": 16, "lr": 0.01, "aux": 3.0},
                        )
                    else:
                        trainer.metrics_processor.log.assert_not_called()
                    self.assertEqual(next(iterator).index, 4)


if __name__ == "__main__":
    unittest.main()
