# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MTP integration with the existing Loop entry, validation and metrics."""

import copy
import random
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torchtitan.experiments.loop_qwen35.loss import LoopMTPLoss
from torchtitan.experiments.loop_qwen35.trainer import SFTTrainer
from torchtitan.experiments.loop_qwen35.train import parse_args


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("resumed", [False, True])
def test_initial_validation_runs_after_loading_weights(enabled, resumed):
    from torchtitan.experiments.loop_qwen35.engine import LoopTrainer

    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.engine = SimpleNamespace(
        mtp_settings={"enabled": True} if enabled else None, num_completed_steps=3
    )
    trainer.experiment_args = SimpleNamespace(eval_steps=3)
    events = []
    trainer.evaluate_fixed = lambda: events.append("validate")

    def native_train(self):
        events.append("load_checkpoint")
        self.on_checkpoint_loaded(resumed)
        events.append("train")

    with patch.object(LoopTrainer, "train", native_train):
        trainer.train()
    assert events == ["load_checkpoint", "validate", "train"]


@pytest.mark.parametrize(
    "option",
    [
        ["--rlhf-type", "dpo"],
        ["--trainable-loop-only", "true"],
        ["--tp", "2"],
    ],
)
def test_entry_rejects_unsupported_mtp_combinations(option):
    argv = [
        "--model",
        "/model",
        "--cached-dataset",
        "/data",
        "--output-dir",
        "/output",
        "--mtp-num-layers",
        "1",
        "--max-length",
        "512",
        "--loss-chunks",
        "4",
    ]
    with patch.dict("os.environ", {"WORLD_SIZE": "4"}), pytest.raises(SystemExit):
        parse_args(argv + option)


@pytest.mark.parametrize("mode", ["mtp_only", "joint"])
@pytest.mark.parametrize("empty", [False, True])
def test_fixed_validation_uses_separate_global_counts_and_restores_rng(mode, empty):
    torch.manual_seed(42)
    predictions = [(torch.randn(3, 4), torch.randn(3, 4)) for _ in range(2)]
    targets = [
        (torch.tensor([1, 2, -100]), torch.tensor([2, -100, -100])),
        (torch.tensor([-100, 3, -100]), torch.tensor([1, 2, -100])),
    ]
    if empty:
        targets = [
            (main, torch.full_like(auxiliary, -100)) for main, auxiliary in targets
        ]

    class Model(torch.nn.Module):
        def preprocess_inputs(self, batch, **kwargs):
            return batch["index"], targets[batch["index"]], {}

        def forward(self, index):
            random.random()
            np.random.rand()
            torch.rand(1)
            return predictions[index]

    class Batch:
        def __init__(self, index):
            self.index = index

        def to_input_dict(self, device):
            return {"index": self.index}

        def to_loss_kwargs(self, device):
            return {}

    class Loader:
        cursor = 7

        def load_state_dict(self, state):
            self.cursor = state["cursor"]

        def __next__(self):
            value = Batch(self.cursor)
            self.cursor += 1
            return value

    trainer = SFTTrainer.__new__(SFTTrainer)
    model = Model().train()
    trainer.engine = SimpleNamespace(
        device=torch.device("cpu"),
        model_parts=[model],
        parallel_dims=None,
        config=SimpleNamespace(parallelism=None),
        max_num_documents=None,
        preprocess_inputs_kwargs={},
        train_context=nullcontext,
        num_completed_steps=5,
        mtp_settings={"training_mode": mode, "loss_scaling_factor": 0.3},
        loss_fn=LoopMTPLoss.Config(training_mode=mode, loss_scaling_factor=0.3).build(),
    )
    trainer.validation_loader, trainer.validation_start = Loader(), {"cursor": 0}
    trainer.experiment_args = SimpleNamespace(eval_batches=2, max_length=3)
    trainer.metrics_processor, trainer.record = (
        SimpleNamespace(log_validation=Mock()),
        Mock(),
    )
    states = copy.deepcopy(
        (random.getstate(), np.random.get_state(), torch.get_rng_state())
    )
    with (
        patch(
            "torch.cuda.get_rng_state",
            return_value=torch.tensor([42], dtype=torch.uint8),
        ),
        patch("torch.cuda.set_rng_state") as restore_cuda,
        patch("torch.distributed.all_reduce"),
    ):
        trainer.evaluate_fixed()
    sums = [
        sum(
            F.cross_entropy(pred[axis], target[axis], reduction="sum")
            for pred, target in zip(predictions, targets)
        )
        for axis in (0, 1)
    ]
    counts = [
        sum(int((target[axis] != -100).sum()) for target in targets) for axis in (0, 1)
    ]
    expected = float(0.3 * sums[1] / max(1, counts[1]))
    if mode == "joint":
        expected += float(sums[0] / counts[0])
    values = trainer.metrics_processor.log_validation.call_args.kwargs
    assert values["loss"] == pytest.approx(expected, abs=1e-6)
    assert values["extra_metrics"]["validation_metrics/main_valid_tokens"] == counts[0]
    assert values["extra_metrics"]["validation_metrics/mtp_valid_tokens"] == counts[1]
    assert trainer.engine.num_completed_steps == 5 and model.training
    assert trainer.validation_loader.cursor == 2
    assert states[0] == random.getstate()
    np.testing.assert_equal(states[1], np.random.get_state())
    assert torch.equal(states[2], torch.get_rng_state())
    restore_cuda.assert_called_once()


def test_joint_mtp_gradient_norm_excludes_backbone_and_does_not_clip():
    from test_mtp_engine import small_engine

    engine, model, _, _ = small_engine("joint")
    for name, parameter in model.named_parameters():
        parameter.grad = torch.full_like(
            parameter, 1.0 if name.startswith("mtp_layers.") else 100.0
        )
    before = [parameter.grad.clone() for parameter in model.parameters()]
    expected = (
        sum(parameter.numel() for parameter in model.mtp_layers.parameters()) ** 0.5
    )
    assert float(engine._mtp_gradient_norm()) == pytest.approx(expected)
    assert all(
        torch.equal(parameter.grad, saved)
        for parameter, saved in zip(model.parameters(), before)
    )


@pytest.mark.parametrize("mode", ["mtp_only", "joint"])
def test_real_entry_optimizer_matches_mtp_trainability(tmp_path, monkeypatch, mode):
    from torch import nn
    from torchtitan.experiments.loop_qwen35 import train as entry
    from torchtitan.experiments.loop_qwen35.model_config import (
        _native_architecture,
        resolve_mtp_config,
    )
    from torchtitan.experiments.loop_qwen35.models.qwen35_loop import (
        LoopSettings,
        model_registry,
    )
    import torchtitan.experiments.loop_qwen35.trainer as sft

    settings = LoopSettings(start=0, end=4, counts=[2], probabilities=[1.0])
    base = model_registry("9B", seq_len=128, loop_settings=settings).model
    resolved = dict(
        model_path=str(tmp_path),
        model_flavor="9B",
        loop_sources=[],
        base_model=_native_architecture(base),
        loop_settings=settings.to_dict(),
        mtp=resolve_mtp_config(num_layers=1, training_mode=mode),
    )
    monkeypatch.setattr(entry, "resolve_model_config", lambda *a, **kw: resolved)
    monkeypatch.setenv("WORLD_SIZE", "1")
    captured = []

    class Trainer:
        def __init__(self, config, **kwargs):
            model = nn.Module()
            layer = nn.Module()
            layer.mhc_attn = nn.Linear(2, 2)
            layer.backbone = nn.Linear(2, 2)
            model.layers = nn.ModuleList([layer])
            model.mtp_layers = nn.ModuleList([nn.Linear(2, 2)])
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(
                    mode == "joint" or name.startswith("mtp_layers.")
                )
            optimizer = config.optimizer.build(model_parts=[model])
            names = [
                name
                for opt in optimizer
                for group in opt.param_groups
                for name in group["param_names"]
            ]
            assert set(names) == {
                n for n, p in model.named_parameters() if p.requires_grad
            }
            assert len(names) == len(set(names))
            schedule = config.lr_scheduler.build(
                optimizers=optimizer, training_steps=config.training.steps
            )
            assert ("lr/router" in schedule.get_metrics()) == (mode == "joint")
            captured.append(config)

        def train(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sft, "SFTTrainer", Trainer)
    entry.main(
        [
            "--model",
            str(tmp_path),
            "--dataset",
            "unused",
            "--output-dir",
            str(tmp_path / "out"),
            "--mtp-num-layers",
            "1",
            "--mtp-training-mode",
            mode,
            "--num-layers",
            "4",
            "--mhc",
            "true",
            "--mhc-start",
            "0",
            "--mhc-end",
            "4",
            "--cp",
            "1",
            "--max-length",
            "128",
            "--loss-chunks",
            "4",
            "--train-iters",
            "10",
            "--router-warmup-steps",
            "0",
            "--gdn-output-kernel",
            "native",
            "--flash-attention-impl",
            "FA2",
        ]
    )
    assert len(captured) == 1
