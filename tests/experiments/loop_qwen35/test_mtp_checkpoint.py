# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Native checkpoint contracts on CPU; CUDA replay is covered by mtp_probe."""

import copy
import json
import numpy as np
import pytest
import random
import torch
import torch.distributed as dist
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.config import TrainingConfig
from torchtitan.training_engine import TrainingEngine
from types import SimpleNamespace
from unittest.mock import patch

from torchtitan.experiments.loop_qwen35.checkpoint import (
    LoopCheckpoint,
    normalized_mtp_metadata,
    select_native_checkpoint,
)
from torchtitan.experiments.loop_qwen35.engine import LoopTrainingEngine
from torchtitan.experiments.loop_qwen35.model_config import resolve_mtp_config
from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import (
    MTP_HF_TO_NATIVE,
    validate_mtp_weights,
)
from mtp_test_utils import state_fingerprint
from test_mtp_weights import source_fixture


class Cursor:
    def __init__(self):
        self.value = 0

    def state_dict(self):
        return {"cursor": self.value}

    def load_state_dict(self, state):
        self.value = state["cursor"]


@pytest.fixture
def factory(tmp_path):
    torch.set_num_threads(1)
    store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    base, mtp_config, weights = source_fixture()
    base.vocab_size = base.tok_embeddings.num_embeddings = base.lm_head.out_features = (
        128
    )
    settings = resolve_mtp_config(num_layers=1)
    settings.update(
        weight_format="qwen3_5_hf_bf16",
        weight_format_version=1,
        source_manifest={
            "source_path": "/synthetic-cpu-fixture",
            "tensors": validate_mtp_weights(weights, mtp_config),
        },
    )
    base.mtp = settings

    def build(*, resume=False, construct=True, disabled=False, streams=0):
        torch.manual_seed(42)
        model_config = copy.deepcopy(base)
        if disabled:
            model_config.mtp = resolve_mtp_config(num_layers=0)
            model_config.mtp_layers = []
        if streams:
            from torchtitan.config import ParallelismConfig

            model_config.mhc = dict(
                start=2,
                end=4,
                streams=streams,
                iterations=20,
                fixed_identity=False,
                pre_scale=1.0,
            )
            model_config.update_from_config(
                config=SimpleNamespace(parallelism=ParallelismConfig())
            )
        model = model_config.build()
        model.init_weights()
        if not disabled:
            model.mtp_layers[0].load_state_dict(
                {
                    native.removeprefix("mtp_layers.0."): weights[source].float()
                    for source, native in MTP_HF_TO_NATIVE.items()
                }
            )
        engine = object.__new__(LoopTrainingEngine)
        engine.model_parts = [model]
        optimizer = default_adamw(lr=0.001, weight_decay=0.2)
        optimizer.implementation = "foreach"
        resolved = {
            "mtp": model_config.mtp,
            "base_model": {"synthetic_cpu": True},
            "loop_settings": model.loop_settings.to_dict(),
        }
        checkpoint = LoopCheckpoint.Config(
            resolved_config=resolved,
            resume=resume,
            interval=1,
            initial_load_path="/must-not-be-read",
        )
        engine.config = SimpleNamespace(
            optimizer=optimizer,
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0, min_lr_factor=0.2
            ),
            training=TrainingConfig(steps=5, disable_cuda_graphs=True),
            checkpointer=checkpoint,
        )
        engine.parallel_dims = SimpleNamespace(
            tp=1,
            cp=1,
            pp=1,
            dp_replicate=1,
            dp_shard=1,
            pp_enabled=False,
            ep_enabled=False,
            get_optional_mesh=lambda name: None,
        )
        engine.device = torch.device("cpu")
        engine.gc_handler, engine.sdc_replayer = (
            SimpleNamespace(run=lambda step: None),
            None,
        )
        engine.output_dir = str(tmp_path)
        engine.num_completed_steps = engine.ntokens_seen = 0
        engine.optimizer_updates = engine.skipped_empty_mtp_steps = 0
        engine._num_optimizer_steps_since_cuda_graph_init = 0
        cursor = Cursor()
        if construct:
            engine._initialize_optimizer(SimpleNamespace(post_optimizer_build_fn=None))
            engine.checkpointer = LoopCheckpoint(
                checkpoint,
                model_parts=[model],
                optimizers=engine.optimizers,
                lr_schedulers=engine.lr_schedulers,
                dataloader=cursor,
                states={"train_state": engine},
                sd_adapter=None,
                base_folder=tmp_path,
            )
        return engine, cursor

    try:
        yield build, tmp_path
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("streams", [2, 4])
def test_mhc_and_mtp_resume_preserves_routers_and_rejects_changed_streams(
    factory, streams
):
    build, _ = factory
    engine, cursor = build(streams=streams)
    update(engine, cursor)
    save(engine)
    expected = state_fingerprint(engine.model_parts[0].state_dict())
    restored, _ = build(resume=True, streams=streams)
    restore(restored)
    assert state_fingerprint(restored.model_parts[0].state_dict()) == expected
    restored.model_parts[0].audit_mtp_trainability()
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for name, parameter in restored.model_parts[0].named_parameters()
        if ".mhc_" in name
    )
    with patch.object(TrainingEngine, "_initialize_optimizer") as initialize:
        with pytest.raises(ValueError, match="Loop configuration"):
            build(resume=True, streams=6 - streams)
        initialize.assert_not_called()


def update(engine, cursor, *, empty=False):
    engine.mtp_global_valid_tokens = torch.tensor(0 if empty else 8)
    engine.prepare_step(torch.tensor(10))
    engine.loss_is_finite = torch.ones((), dtype=torch.int32)
    # An independent analytic objective tests state replay without CUDA kernels.
    coefficient = (
        random.random() + float(np.random.random()) + float(torch.rand(()))
    ) * 0.001
    loss = sum(
        p.square().sum() for p in engine.model_parts[0].parameters() if p.requires_grad
    )
    (loss * (0 if empty else coefficient)).backward()
    gradients = {
        name: p.grad.clone()
        for name, p in engine.model_parts[0].named_parameters()
        if p.requires_grad
    }
    engine.optimizer_step()
    engine.ntokens_seen += 16
    cursor.value += 1
    return gradients


def fingerprint(engine, cursor):
    return state_fingerprint(
        (
            [part.state_dict() for part in engine.model_parts],
            engine.optimizers.state_dict(),
            [s.state_dict() for s in engine.lr_schedulers],
            engine.state_dict(),
            cursor.state_dict(),
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
    )


def save(engine):
    with patch("torch.cuda.get_rng_state", side_effect=torch.get_rng_state):
        assert engine.checkpointer.save(engine.num_completed_steps, last_step=True)


def restore(engine):
    with patch.object(
        engine.checkpointer, "_load_hf", side_effect=AssertionError("HF overwrite")
    ):
        with patch("torch.cuda.set_rng_state"):
            assert engine.load_checkpoint()


@pytest.mark.parametrize("empty", [False, True])
def test_three_steps_equal_two_plus_fresh_restore(factory, empty):
    build, folder = factory
    random.seed(42)
    np.random.seed(42)
    continuous, cursor = build()
    update(continuous, cursor)
    update(continuous, cursor, empty=empty)
    save(continuous)
    saved = fingerprint(continuous, cursor)
    expected_gradients = update(continuous, cursor)
    expected = fingerprint(continuous, cursor)
    resumed, resumed_cursor = build(resume=True)
    restore(resumed)
    assert fingerprint(resumed, resumed_cursor) == saved
    actual_gradients = update(resumed, resumed_cursor)
    assert state_fingerprint(actual_gradients) == state_fingerprint(expected_gradients)
    assert fingerprint(resumed, resumed_cursor) == expected
    assert resumed.optimizer_updates == 3 - int(empty)
    assert resumed.skipped_empty_mtp_steps == int(empty)
    assert not list((folder / "checkpoints").rglob("*.tmp"))


@pytest.mark.parametrize(
    "failure",
    [
        "mode",
        "coefficient",
        "loop",
        "topology",
        "base",
        "frozen",
        "groups",
        "optimizer",
        "scheduler",
        "source",
        "version",
        "missing_mtp",
    ],
)
def test_metadata_rejection_precedes_optimizer_construction(factory, failure):
    build, folder = factory
    engine, cursor = build()
    update(engine, cursor)
    save(engine)
    marker = folder / "checkpoints/step-1/complete.json"
    metadata = json.loads(marker.read_text())
    mtp = metadata["mtp"]
    if failure == "mode":
        mtp["training_mode"] = "joint"
    elif failure == "coefficient":
        mtp["loss_scaling_factor"] = 0.3
    elif failure == "loop":
        metadata["topology"]["loop"]["start"] += 1
    elif failure == "topology":
        metadata["topology"]["tp"] = 2
    elif failure == "base":
        metadata["resolved_config"]["base_model"] = {"different": True}
    elif failure == "frozen":
        mtp["frozen_parameters"] = mtp["frozen_parameters"][1:]
    elif failure == "groups":
        mtp["optimizer_groups"][0]["parameters"][0].pop()
    elif failure == "optimizer":
        mtp["optimizer_config"] = {}
    elif failure == "scheduler":
        mtp["scheduler_config"]["min_lr_factor"] = 0.9
    elif failure == "source":
        mtp["source_manifest"]["tensors"].pop(next(iter(MTP_HF_TO_NATIVE)))
    elif failure == "version":
        mtp["version"] = 100
    else:
        del metadata["mtp"]
    marker.write_text(json.dumps(metadata))
    resumed, _ = build(resume=True, construct=False)
    with patch.object(TrainingEngine, "_initialize_optimizer") as constructor:
        with pytest.raises(ValueError):
            resumed._initialize_optimizer(SimpleNamespace(post_optimizer_build_fn=None))
        constructor.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        "model_mtp",
        "model_backbone",
        "optimizer",
        "scheduler",
        "counters",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "dataloader",
    ],
)
def test_missing_payload_rejected_before_weight_assignment(factory, failure):
    build, folder = factory
    engine, cursor = build()
    update(engine, cursor)
    save(engine)
    path = folder / "checkpoints/step-1/rank-0.pt"
    state = torch.load(path, weights_only=False)
    if failure.startswith("model_"):
        key = (
            "mtp_layers.0.enorm.weight"
            if failure == "model_mtp"
            else "tok_embeddings.weight"
        )
        state["model"][0].pop(key)
    elif failure == "optimizer":
        state["optimizer"].pop(next(iter(state["optimizer"])))
    elif failure == "scheduler":
        state["mtp_scheduler_states"][0].pop("_last_lr")
    elif failure == "counters":
        state["train_state"]["optimizer_updates"] = 9
    else:
        del state[failure]
    torch.save(state, path)
    resumed, _ = build(resume=True)
    before = state_fingerprint(resumed.model_parts[0].state_dict())
    with pytest.raises(ValueError):
        restore(resumed)
    assert state_fingerprint(resumed.model_parts[0].state_dict()) == before


def test_legacy_disabled_restore_and_incomplete_selection(factory):
    build, folder = factory
    engine, cursor = build(disabled=True)
    update(engine, cursor)
    save(engine)
    marker = folder / "checkpoints/step-1/complete.json"
    metadata = json.loads(marker.read_text())
    metadata.pop("mtp")
    metadata["resolved_config"].pop("mtp")
    marker.write_text(json.dumps(metadata))
    assert normalized_mtp_metadata(metadata) == {"enabled": False}
    resumed, resumed_cursor = build(resume=True, disabled=True)
    restore(resumed)
    assert state_fingerprint(resumed.model_parts[0].state_dict()) == state_fingerprint(
        engine.model_parts[0].state_dict()
    )
    assert resumed_cursor.value == 1 and resumed.num_completed_steps == 1
    (folder / "checkpoints/step-99").mkdir()
    assert (
        select_native_checkpoint(folder / "checkpoints", resume=True).name == "step-1"
    )
    assert select_native_checkpoint(folder / "checkpoints", resume=False) is None
    with pytest.raises(ValueError, match="incomplete"):
        select_native_checkpoint(folder / "checkpoints", resume=True, step=99)
    with pytest.raises(ValueError, match="enable or disable"):
        build(resume=True)


def test_hf_fresh_is_not_native_restore(factory):
    build, _ = factory
    engine, cursor = build()
    with patch.object(engine.checkpointer, "_load_hf") as load_hf:
        with patch.object(engine.optimizers, "load_state_dict") as load_optimizer:
            assert engine.load_checkpoint() is False
    load_hf.assert_called_once_with()
    load_optimizer.assert_not_called()
    assert cursor.value == engine.num_completed_steps == engine.optimizer_updates == 0
