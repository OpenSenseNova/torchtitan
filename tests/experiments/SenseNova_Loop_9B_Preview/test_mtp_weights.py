# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Strict source import and declared-placement reconstruction contracts."""

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.protocols.sharding import ShardingConfig
from types import SimpleNamespace
from unittest.mock import Mock, patch

from torchtitan.experiments.SenseNova_Loop_9B_Preview.checkpoint import (
    LoopCheckpoint,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
    _load_loop_module,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.models.qwen35_mtp import (
    MTP_HF_TO_NATIVE,
    build_mtp_config,
    slice_mtp_for_tp,
    validate_mtp_weights,
)


def source_fixture():
    loop = _load_loop_module()
    base = loop.model_registry(
        "debugmodel", seq_len=16, loop_settings=loop.LoopSettings(start=2, end=4)
    ).model
    config = build_mtp_config(base)
    dim = base.dim
    attn = config.attention
    hidden = config.feed_forward.w2.in_features
    # Independent list of the HF checkpoint's public tensor contract.
    shapes = {
        "pre_fc_norm_embedding": (dim,),
        "pre_fc_norm_hidden": (dim,),
        "fc": (dim, 2 * dim),
        "norm": (dim,),
        "layers.0.self_attn.q_proj": (2 * attn.n_heads * attn.head_dim, dim),
        "layers.0.self_attn.k_proj": (attn.n_kv_heads * attn.head_dim, dim),
        "layers.0.self_attn.v_proj": (attn.n_kv_heads * attn.head_dim, dim),
        "layers.0.self_attn.o_proj": (dim, attn.n_heads * attn.head_dim),
        "layers.0.self_attn.q_norm": (attn.head_dim,),
        "layers.0.self_attn.k_norm": (attn.head_dim,),
        "layers.0.mlp.gate_proj": (hidden, dim),
        "layers.0.mlp.up_proj": (hidden, dim),
        "layers.0.mlp.down_proj": (dim, hidden),
        "layers.0.input_layernorm": (dim,),
        "layers.0.post_attention_layernorm": (dim,),
    }
    generator = torch.Generator().manual_seed(42)
    weights = {
        f"mtp.{key}.weight": torch.randn(shape, generator=generator).to(torch.bfloat16)
        for key, shape in shapes.items()
    }
    base.mtp = {"enabled": True}
    base.mtp_layers = [config]
    return base, config, weights


def test_all_source_keys_exact_roundtrip_and_independent_head():
    base, config, weights = source_fixture()
    assert len(weights) == 15 and weights.keys() == MTP_HF_TO_NATIVE.keys()
    adapter = _load_loop_module().Qwen35LoopStateDictAdapter(base, hf_assets_path=None)
    embedding = torch.ones(8, base.dim, dtype=torch.bfloat16)
    head = embedding + 2
    weights.update(
        {"model.language_model.embed_tokens.weight": embedding, "lm_head.weight": head}
    )
    state = adapter.from_hf(weights)
    assert (
        state["lm_head.weight"] is head and state["tok_embeddings.weight"] is embedding
    )
    with torch.device("meta"):
        block = config.build()
    block.load_state_dict(
        {
            key.removeprefix("mtp_layers.0."): value
            for key, value in state.items()
            if key.startswith("mtp_layers.")
        },
        assign=True,
        strict=True,
    )
    assert len(list(block.parameters())) == 14 and len(block.state_dict()) == 15
    exported = adapter.to_hf(
        {f"mtp_layers.0.{key}": value for key, value in block.state_dict().items()}
    )
    for key in weights:
        if key.startswith("mtp."):
            assert torch.equal(weights[key], exported[key])
            assert exported[key].dtype == torch.bfloat16
    fused = block.feed_forward.w13.weight.unflatten(0, (-1, 2))
    assert torch.equal(fused[:, 0], weights["mtp.layers.0.mlp.gate_proj.weight"])
    assert torch.equal(fused[:, 1], weights["mtp.layers.0.mlp.up_proj.weight"])


@pytest.mark.parametrize("streams", [2, 4])
def test_hf_roundtrip_preserves_mhc_gate_and_all_mtp_tensors(streams):
    from torchtitan.config import ParallelismConfig
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.model_config import (
        resolve_mtp_config,
    )

    base, _, weights = source_fixture()
    base.mtp = resolve_mtp_config(num_layers=1)
    base.vocab_size = (
        base.tok_embeddings.num_embeddings
    ) = base.lm_head.out_features = 128
    base.mhc = dict(
        start=2, end=4, streams=streams, iterations=20, fixed_identity=False
    )
    base.update_from_config(config=SimpleNamespace(parallelism=ParallelismConfig()))
    with torch.device("meta"):
        model = base.build()
    native = {target: weights[source] for source, target in MTP_HF_TO_NATIVE.items()}
    native.update(
        {
            name: torch.randn(128, base.dim, dtype=torch.bfloat16)
            for name in ("tok_embeddings.weight", "lm_head.weight")
        }
    )
    native.update(
        {
            name: torch.randn(value.shape)
            for name, value in model.state_dict().items()
            if ".mhc_" in name or name == "loop_gate.logits"
        }
    )
    assert any(".mhc_" in name for name in native) and "loop_gate.logits" in native
    adapter = _load_loop_module().Qwen35LoopStateDictAdapter(base, hf_assets_path=None)
    restored = adapter.from_hf(adapter.to_hf(native))
    assert restored.keys() == native.keys()
    assert all(torch.equal(value, restored[name]) for name, value in native.items())
    model.audit_mtp_trainability()


@pytest.mark.parametrize(
    "failure",
    ["all_missing", "one_missing", "extra_layer", "shape", "dtype", "nan", "inf"],
)
def test_reject_invalid_weights_before_assignment(failure):
    _, config, weights = source_fixture()
    key = "mtp.fc.weight"
    if failure == "all_missing":
        weights = {}
    elif failure == "one_missing":
        weights.pop(key)
    elif failure == "extra_layer":
        weights["mtp.layers.1.input_layernorm.weight"] = weights["mtp.norm.weight"]
    elif failure == "shape":
        weights[key] = weights[key][:-1]
    elif failure == "dtype":
        weights[key] = weights[key].float()
    else:
        weights[key][0, 0] = float(failure)
    with pytest.raises(ValueError):
        validate_mtp_weights(weights, config)


def test_disabled_ignores_source_mtp_even_if_malformed():
    base, _, weights = source_fixture()
    base.mtp = {"enabled": False}
    weights["mtp.fc.weight"] = torch.tensor(float("nan"))
    weights["lm_head.weight"] = torch.ones(8, base.dim, dtype=torch.bfloat16)
    adapter = _load_loop_module().Qwen35LoopStateDictAdapter(base, hf_assets_path=None)
    assert not any(key.startswith("mtp_layers.") for key in adapter.from_hf(weights))


def _tp_reassemble_worker(rank, rendezvous):
    store = dist.TCPStore("127.0.0.1", rendezvous, is_master=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        _, config, weights = source_fixture()
        block = config.build()
        model = nn.Module()
        model.mtp_layers = nn.ModuleList([block])
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) or hasattr(module, "weight"):
                axis = (
                    0
                    if name.endswith((".wq", ".wk", ".wv", ".w13"))
                    else 1
                    if name.endswith((".wo", ".w2"))
                    else None
                )
                module._sharding_config = ShardingConfig(
                    state_shardings={
                        "weight": dense_param_placement(
                            tp=spmd.R if axis is None else spmd.S(axis)
                        )
                    }
                )
        for source, target in MTP_HF_TO_NATIVE.items():
            value = weights[source]
            local = slice_mtp_for_tp(model, target, value, mesh)
            pieces = [torch.empty_like(local) for _ in range(2)]
            dist.all_gather(pieces, local)
            if any(
                fragment in target
                for fragment in (".wq.", ".wk.", ".wv.", ".w1.", ".w3.")
            ):
                reconstructed = torch.cat(pieces, dim=0)
            elif any(fragment in target for fragment in (".wo.", ".w2.")):
                reconstructed = torch.cat(pieces, dim=1)
            else:
                assert torch.equal(pieces[0], pieces[1])
                reconstructed = pieces[0]
            assert torch.equal(reconstructed, value), source
        block.enorm._sharding_config = None
        with pytest.raises(ValueError, match="Missing declared"):
            slice_mtp_for_tp(
                model,
                "mtp_layers.0.enorm.weight",
                weights["mtp.pre_fc_norm_embedding.weight"],
                mesh,
            )
    finally:
        dist.destroy_process_group()


def test_declared_tp_reassembly_all_15_tensors():
    store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    mp.spawn(_tp_reassemble_worker, args=(store.port,), nprocs=2, join=True)


def test_hf_initialization_is_fresh_not_resume(tmp_path):
    checkpoint = object.__new__(LoopCheckpoint)
    checkpoint.config = LoopCheckpoint.Config(resume=False, initial_step=0)
    checkpoint.folder = tmp_path
    checkpoint.engine = SimpleNamespace(num_completed_steps=999)
    checkpoint.optimizers, checkpoint.schedulers, checkpoint.dataloader = (
        Mock(),
        Mock(),
        Mock(),
    )
    checkpoint._load_hf, checkpoint._save_resolved_config = Mock(), Mock()
    assert checkpoint.load() is False
    assert checkpoint.engine.num_completed_steps == 0
    checkpoint._load_hf.assert_called_once_with()
    for stateful in (
        checkpoint.optimizers,
        checkpoint.schedulers,
        checkpoint.dataloader,
    ):
        stateful.load_state_dict.assert_not_called()


def test_untied_missing_head_fails_before_model_load(tmp_path):
    checkpoint = object.__new__(LoopCheckpoint)
    checkpoint.config = LoopCheckpoint.Config(
        initial_load_path=str(tmp_path),
        resolved_config={"base_model": {"tie_word_embeddings": False}},
    )
    checkpoint.adapter = SimpleNamespace(
        hf_language_model_prefix="model.language_model"
    )
    checkpoint.parts = [SimpleNamespace(config=SimpleNamespace(layers=[None]))]
    with patch(
        "torchtitan.experiments.SenseNova_Loop_9B_Preview.checkpoint.load_file",
        return_value={"model.language_model.embed_tokens.weight": torch.ones(4, 2)},
    ):
        with pytest.raises(KeyError, match="lm_head"):
            checkpoint._load_hf()
