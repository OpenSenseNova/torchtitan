# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from torchtitan.config import FSDPSymmMemScope
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointing
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, enable_fsdp_symm_mem


class _FSDPModule(nn.Module):
    def __init__(self, *, moe_enabled: bool) -> None:
        super().__init__()
        self.moe_enabled = moe_enabled
        self.force_sum_enabled = False
        self.symm_mem_enabled = False

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        self.force_sum_enabled = enable

    def set_symm_mem_for_comm(self) -> None:
        self.symm_mem_enabled = True


@pytest.mark.parametrize(
    ("scope", "dense_enabled", "sparse_enabled"),
    [
        (None, False, False),
        ("all", True, True),
        ("dense", True, False),
    ],
)
def test_enable_fsdp_symm_mem_scope(
    scope: FSDPSymmMemScope, dense_enabled: bool, sparse_enabled: bool
) -> None:
    model = nn.Module()
    dense = _FSDPModule(moe_enabled=False)
    sparse = _FSDPModule(moe_enabled=True)
    model.add_module("dense", dense)
    model.add_module("sparse", sparse)

    with mock.patch("torchtitan.distributed.fsdp.FSDPModule", _FSDPModule):
        enable_fsdp_symm_mem(model, scope)

    assert dense.force_sum_enabled == dense_enabled
    assert dense.symm_mem_enabled == dense_enabled
    assert sparse.force_sum_enabled == sparse_enabled
    assert sparse.symm_mem_enabled == sparse_enabled


def _decoder():
    model = nn.Module()
    model.enable_weight_tying = False
    model.tok_embeddings = nn.Embedding(8, 4)
    model.norm = nn.LayerNorm(4)
    model.lm_head = nn.Linear(4, 8, bias=False)
    model.layers = nn.ModuleDict({"0": nn.Linear(4, 4), "1": nn.Linear(4, 4)})
    model.mtp_layers = nn.ModuleList([nn.Linear(4, 4)])
    return model


@pytest.mark.parametrize("extra", [(), ("mtp_layers",)])
def test_extra_fsdp_containers_preserve_registration_and_order(extra):
    model = _decoder()
    before = tuple(model.state_dict())
    calls = []

    def shard(module, **kwargs):
        assert tuple(model.layers) == ("0", "1")
        assert tuple(model.state_dict()) == before
        calls.append(module)

    with mock.patch("torchtitan.distributed.fsdp.fully_shard", side_effect=shard):
        apply_fsdp_to_decoder(
            model,
            SimpleNamespace(mesh_dim_names=("dp_shard",)),
            torch.bfloat16,
            torch.float32,
            False,
            extra_layer_names=extra,
        )
    expected = [
        model.tok_embeddings,
        [model.norm, model.lm_head],
        *model.layers.values(),
    ]
    if extra:
        expected += list(model.mtp_layers)
    assert calls == expected + [model]
    assert tuple(model.state_dict()) == before
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})


@pytest.mark.parametrize("extra", [(), ("mtp_layers",)])
def test_extra_ac_containers_use_stable_names_and_parameter_identities(extra):
    model = _decoder()
    wrapped = []

    class RecordingAC(ActivationCheckpointing):
        def _wrap_block(self, module, *, base_fqn=None):
            wrapped.append((base_fqn, module))
            return module

    identities = tuple(id(p) for p in model.parameters())
    RecordingAC(ActivationCheckpointing.Config()).apply(model, extra_layer_names=extra)
    expected = [(f"layers.{key}", layer) for key, layer in model.layers.items()]
    if extra:
        expected += [("mtp_layers.0", model.mtp_layers[0])]
    assert wrapped == expected
    assert tuple(id(p) for p in model.parameters()) == identities


def test_repeated_layer_rejected_before_fsdp_mutation():
    model = _decoder()
    model.mtp_layers[0] = model.layers["0"]
    with mock.patch("torchtitan.distributed.fsdp.fully_shard") as shard:
        with pytest.raises(ValueError, match="same block twice"):
            apply_fsdp_to_decoder(
                model,
                SimpleNamespace(mesh_dim_names=("dp_shard",)),
                torch.bfloat16,
                torch.float32,
                False,
                extra_layer_names=("mtp_layers",),
            )
    shard.assert_not_called()


def test_deepseek_mtp_fsdp_keeps_original_container():
    from torchtitan.models.deepseek_v3.mtp import apply_fsdp_to_mtp_decoder

    model = _decoder()

    def apply(model, mesh, **kwargs):
        assert tuple(model.layers) == ("0", "1")
        assert kwargs["extra_layer_names"] == ("mtp_layers",)

    with mock.patch(
        "torchtitan.models.deepseek_v3.mtp.apply_fsdp_to_decoder", side_effect=apply
    ):
        apply_fsdp_to_mtp_decoder(model, None, torch.bfloat16, torch.float32, False)
