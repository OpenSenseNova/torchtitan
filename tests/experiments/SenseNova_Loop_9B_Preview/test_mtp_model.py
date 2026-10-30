# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU model math, post-Loop execution and frozen-parameter contracts."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.components.loss import ChunkedLossWrapper
from types import SimpleNamespace

from torchtitan.experiments.SenseNova_Loop_9B_Preview.engine import (
    LoopTrainingEngine,
)
from mtp_test_utils import FrozenAudit, logical_mtp_gradients
from mtp_reference import mtp_reference
from test_loop_semantics import AffineLayer, affine_model, loop
from test_mtp_weights import source_fixture


class DenseCPUAttention(nn.Module):
    """Use CPU SDPA only in place of the GPU inner attention kernel."""

    def forward(self, query, key, value, *, attention_masks, scale, enable_gqa):
        result = F.scaled_dot_product_attention(
            query.transpose(0, 1)[None],
            key.transpose(0, 1)[None],
            value.transpose(0, 1)[None],
            attn_mask=attention_masks,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return result[0].transpose(0, 1)


def test_disabled_default_and_explicit_zero_initialization_rng_and_output():
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.model_config import (
        resolve_mtp_config,
    )

    states, rngs = [], []
    for explicit in (False, True):
        config = loop.model_registry(
            "debugmodel", seq_len=16, loop_settings=loop.LoopSettings(start=2, end=4)
        ).model
        config.vocab_size = (
            config.tok_embeddings.num_embeddings
        ) = config.lm_head.out_features = 128
        if explicit:
            config.mtp = resolve_mtp_config(num_layers=0)
        torch.manual_seed(42)
        model = config.build()
        model.init_weights()
        assert not any(name.startswith("mtp_layers.") for name in model.state_dict())
        states.append(
            {name: value.clone() for name, value in model.state_dict().items()}
        )
        rngs.append(torch.get_rng_state())
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(value, states[1][name]) for name, value in states[0].items())
    assert torch.equal(rngs[0], rngs[1])
    results = []
    for settings in (None, resolve_mtp_config(num_layers=0)):
        model = affine_model([AffineLayer(index) for index in range(8)])
        model.mtp_settings = settings
        value = torch.tensor([0.2, -0.1], requires_grad=True)
        output = model(value)
        assert isinstance(output, torch.Tensor)
        results.append(
            (
                output.detach(),
                torch.autograd.grad(
                    output.square().sum(), [value, *model.parameters()]
                ),
            )
        )
    assert torch.equal(results[0][0], results[1][0])
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(results[0][1], results[1][1])
    )


@pytest.mark.parametrize("multimodal_positions", [False, True])
def test_native_mtp_block_matches_independent_reference(multimodal_positions):
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.models.qwen35_mtp import (
        MTP_HF_TO_NATIVE,
    )

    torch.manual_seed(42)
    base, config, source = source_fixture()
    weights = {name: value.float() * 0.02 for name, value in source.items()}
    block = config.build()
    block.attn.inner_attention = DenseCPUAttention()
    block.load_state_dict(
        {
            target.removeprefix("mtp_layers.0."): weights[name]
            for name, target in MTP_HF_TO_NATIVE.items()
        },
        strict=True,
    )
    documents = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    slots = torch.arange(8)
    visible = (documents[:, None] == documents[None, :]) & (
        slots[:, None] >= slots[None, :]
    )
    coordinates = torch.tensor([1, 2, 0, 1, 2, 3, 4, 0])
    positions = (
        torch.stack((coordinates, coordinates // 2, coordinates % 3), -1)
        if multimodal_positions
        else coordinates
    )
    hidden, embedding = torch.randn(8, base.dim), torch.randn(8, base.dim)
    full_rotary = []
    handle = block.attn.inner_attention.register_forward_pre_hook(
        lambda module, inputs: full_rotary.extend(inputs[:2])
    )
    actual = block(hidden, embedding, {"quadratic_attention": visible}, positions)
    handle.remove()
    references = {
        name: value.clone().requires_grad_() for name, value in weights.items()
    }
    attention = config.attention
    expected, intermediates = mtp_reference(
        references,
        hidden,
        embedding,
        positions,
        documents,
        num_heads=attention.n_heads,
        num_kv_heads=attention.n_kv_heads,
        head_dim=attention.head_dim,
        rotary_dim=attention.rotary_dim,
        theta=attention.rope.theta,
        sections=attention.rope.mrope_section,
        return_intermediates=True,
    )
    for name, value in zip(("rotary_query", "rotary_key"), full_rotary):
        assert value.shape[-1] == attention.head_dim
        torch.testing.assert_close(value, intermediates[name], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    gradients = logical_mtp_gradients(block)
    for source, target in MTP_HF_TO_NATIVE.items():
        torch.testing.assert_close(
            gradients[target.removeprefix("mtp_layers.0.")],
            references[source].grad,
            atol=1e-5,
            rtol=1e-4,
        )
    changed = embedding.clone()
    changed[3:] += 5
    isolated = block(hidden, changed, {"quadratic_attention": visible}, positions)
    assert torch.equal(actual[:3], isolated[:3])


def test_real_mtp_block_learns_fixed_targets_in_ten_updates():
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.models.qwen35_mtp import (
        MTP_HF_TO_NATIVE,
    )

    torch.manual_seed(42)
    base, config, source = source_fixture()
    block = config.build()
    block.attn.inner_attention = DenseCPUAttention()
    block.load_state_dict(
        {
            native.removeprefix("mtp_layers.0."): source[name].float() * 0.02
            for name, native in MTP_HF_TO_NATIVE.items()
        }
    )
    hidden, embedding = torch.randn(8, base.dim), torch.randn(8, base.dim)
    head = nn.Linear(base.dim, 16, bias=False).requires_grad_(False)
    labels, positions = torch.arange(8), torch.arange(8)
    masks = {"quadratic_attention": torch.ones(8, 8, dtype=torch.bool).tril()}
    frozen = [value.clone() for value in (hidden, embedding, head.weight)]
    optimizer = torch.optim.AdamW(block.parameters(), lr=0.001, weight_decay=0)
    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(block(hidden, embedding, masks, positions)), labels)
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in block.parameters()
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3 - 0.01, losses
    assert head.weight.grad is None
    assert all(
        torch.equal(value, saved)
        for value, saved in zip((hidden, embedding, head.weight), frozen)
    )


class FusionRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 4, bias=False)
        self.calls = 0

    def forward(self, hidden, embedding, masks, positions, **kwargs):
        self.calls += 1
        self.last_hidden, self.last_embedding = hidden, embedding
        return self.projection(torch.cat((embedding, hidden), dim=-1))


def structural_model(mode="mtp_only", gate=True):
    model = affine_model([AffineLayer(index) for index in range(8)])
    model.tok_embeddings = nn.Embedding(16, 4)
    model.vision_encoder = None
    model.norm = nn.Linear(4, 4, bias=False)
    model.lm_head = nn.Linear(4, 16, bias=False)
    model.mtp_layers = nn.ModuleList([FusionRecorder()])
    model.mtp_input_shard = None
    model.mtp_settings = {"enabled": True, "training_mode": mode}
    model.loop_gate = loop._loop_gate_config(4, model.loop_settings).build()
    model.loop_gate.init_states()
    model.loop_settings.use_gate = gate
    model.configure_mtp_trainability()
    return model


@pytest.mark.parametrize("chunks", [0, 2])
def test_loop_mtp_once_shared_parameters_and_two_frozen_updates(chunks):
    torch.manual_seed(42)
    model = structural_model()
    model.train()
    assert model.training and model.mtp_layers.training
    assert all(
        not child.training
        for name, child in model.named_children()
        if name != "mtp_layers"
    )
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=0.01,
        weight_decay=0,
    )
    engine = object.__new__(LoopTrainingEngine)
    engine.model_parts, engine.optimizers = [model], [optimizer]
    engine._audit_mtp_parameters(optimizer=True)
    audit = FrozenAudit(model)
    before = {
        name: value.detach().clone()
        for name, value in model.mtp_layers.named_parameters()
    }
    modes, main_outputs = [], []
    for layer in model.layers.values():
        layer.register_forward_pre_hook(
            lambda *args: modes.append(torch.is_grad_enabled())
        )
    tokens, future = torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3, 4, 0])
    positions, labels = torch.arange(4), torch.tensor([3, 4, -100, -100])
    embeddings, head = model.tok_embeddings, model.lm_head
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        hidden, auxiliary = model(
            tokens, positions=positions, mtp_input=future, mtp_positions=positions
        )
        assert model.mtp_layers[0].calls == step + 1
        assert model.mtp_layers[0].last_hidden is hidden
        assert torch.equal(model.mtp_layers[0].last_embedding, embeddings(future))
        assert (
            not hidden.requires_grad
            and not torch.is_inference(hidden)
            and auxiliary.requires_grad
        )
        main_outputs.append(head(hidden).detach())
        auxiliary.retain_grad()
        reference = auxiliary.detach().clone().requires_grad_()
        (expected_gradient,) = torch.autograd.grad(
            F.cross_entropy(head(reference), labels), reference
        )
        if chunks:
            wrapper = ChunkedLossWrapper(ChunkedLossWrapper.Config(num_chunks=chunks))
            wrapper.loss_fn = lambda pred, target, count, **kwargs: (
                F.cross_entropy(pred[1], target[1], reduction="sum") / count,
                {},
            )
            wrapper.set_lm_head(head)
            loss, _ = wrapper((hidden, auxiliary), (labels, labels), torch.tensor(2))
        else:
            loss = F.cross_entropy(head(auxiliary), labels)
        loss.backward()
        torch.testing.assert_close(auxiliary.grad, expected_gradient)
        assert torch.count_nonzero(auxiliary.grad)
        optimizer.step()
        engine._audit_mtp_parameters(optimizer=True)
        audit.check(model, [optimizer])
    assert modes and not any(modes)
    assert torch.equal(*main_outputs)
    assert model.tok_embeddings is embeddings and model.lm_head is head
    for name, value in model.mtp_layers.named_parameters():
        assert torch.count_nonzero(value.grad) and not torch.equal(value, before[name])


def test_mtp_receives_final_normalized_hidden_after_complete_loop():
    torch.manual_seed(42)
    model = structural_model(gate=False)
    tokens, positions = torch.tensor([1, 2, 3, 4]), torch.arange(4)
    observed = []
    model.norm.register_forward_hook(
        lambda module, inputs, output: observed.append(output)
    )
    model._skip_lm_head = False
    main, auxiliary = model(
        tokens, positions=positions, mtp_input=tokens, mtp_positions=positions
    )
    assert len(observed) == model.mtp_layers[0].calls == 1
    assert model.mtp_layers[0].last_hidden is observed[0]
    assert torch.equal(main, model.lm_head(observed[0]))
    assert auxiliary.requires_grad and not main.requires_grad


class StructuralVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4, bias=False)
        self.spatial_merge_size = self.spatial_merge_unit = 1
        self.config = SimpleNamespace(patch_embed_proj=SimpleNamespace(in_features=4))
        self.calls = 0
        self.grad_enabled = []

    def forward(self, pixels, *, grid_thw):
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.patch_embed(pixels)


@pytest.mark.parametrize("image", [True, False])
def test_one_vision_bank_shared_with_future_embeddings(image):
    torch.manual_seed(42)
    model = structural_model()
    model.tok_embeddings = nn.Embedding(64, 4)
    model.vision_encoder = StructuralVision()
    model.configure_mtp_trainability()
    model.train()
    audit = FrozenAudit(model)
    tokens = torch.tensor([3, 42, 42, 9]) if image else torch.tensor([3, 7, 11, 9])
    future = torch.cat((tokens[1:], torch.zeros(1, dtype=torch.long)))
    pixels = torch.arange(8).reshape(2, 4).float() / 7 if image else None
    indices = torch.tensor([0, 1, -1, -1]) if image else None
    masks = {"quadratic_attention": torch.ones(4, 4, dtype=torch.bool).tril()}
    main, auxiliary = model(
        tokens,
        positions=torch.arange(4),
        attention_masks=masks,
        mtp_input=future,
        mtp_positions=torch.tensor([1, 2, 3, 0]),
        mtp_attention_masks=masks,
        mtp_vision_bank_indices=indices,
        pixel_values=pixels,
        grid_thw=torch.tensor([[1, 1, 2]]) if image else None,
        special_tokens={"image_id": 42, "video_id": 43},
    )
    assert model.vision_encoder.calls == 1 and model.vision_encoder.grad_enabled == [
        False
    ]
    expected = model.tok_embeddings(future)
    if image:
        expected[:2] = F.linear(pixels, model.vision_encoder.patch_embed.weight)
    assert torch.equal(model.mtp_layers[0].last_embedding, expected)
    optimizer = torch.optim.AdamW(
        model.mtp_layers.parameters(), lr=0.01, weight_decay=0
    )
    F.cross_entropy(
        model.lm_head(auxiliary), torch.tensor([4, 5, -100, -100])
    ).backward()
    optimizer.step()
    audit.check(model, [optimizer])
    assert not main.requires_grad


@pytest.mark.parametrize("length", [1, 2])
def test_invalid_tail_slots_keep_finite_attention_and_gradients(length):
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.models.qwen35_mtp import (
        MTP_HF_TO_NATIVE,
    )

    base, config, weights = source_fixture()
    block = config.build()
    block.attn.inner_attention = DenseCPUAttention()
    block.load_state_dict(
        {
            target.removeprefix("mtp_layers.0."): weights[name].float() * 0.02
            for name, target in MTP_HF_TO_NATIVE.items()
        },
        strict=True,
    )
    # Every slot can attend to itself, even when no MTP loss target is valid.
    output = block(
        torch.zeros(length, base.dim),
        torch.zeros(length, base.dim),
        {"quadratic_attention": torch.ones(length, length, dtype=torch.bool).tril()},
        torch.zeros(length, 3, dtype=torch.long),
    )
    assert torch.isfinite(output).all()
    (output.sum() * 0).backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in block.parameters()
    )


@pytest.mark.parametrize("skip_dp", [False, True])
def test_parallelizer_wraps_mtp_once_and_recomputes_only_trainable_block(skip_dp):
    from torchtitan.config import ParallelismConfig, TrainingConfig
    from torchtitan.distributed.activation_checkpoint import FullAC
    from unittest.mock import patch

    torch.manual_seed(42)
    model = structural_model(gate=False)
    model.config.freeze_vision = True
    model.config.trainable_loop_only = False
    model.config.fsdp_prefetch = False
    model.enable_weight_tying = False
    model.parallelize = lambda dims: None
    original_parameters = tuple(model.parameters())
    requires_grad = {id(p): p.requires_grad for p in original_parameters}
    calls = []
    for key, layer in model.layers.items():
        layer.register_forward_pre_hook(lambda *args, key=key: calls.append(key))
    mtp = model.mtp_layers[0]
    head = model.lm_head
    mesh = SimpleNamespace(mesh_dim_names=("loop_fsdp",))
    dims = SimpleNamespace(
        ep_enabled=False,
        cp_enabled=False,
        pp_enabled=False,
        dp_replicate_enabled=False,
        get_optional_mesh=lambda *a, **k: SimpleNamespace(_flatten=lambda name: mesh),
    )
    sharded = []
    with patch(
        "torchtitan.distributed.fsdp.fully_shard",
        side_effect=lambda mod, **kw: sharded.append(mod),
    ):
        returned = loop.parallelize_qwen35_loop(
            model,
            skip_dp=skip_dp,
            parallel_dims=dims,
            training=TrainingConfig(),
            parallelism=ParallelismConfig(),
            compile_config=None,
            ac_config=FullAC.Config(preserve_rng_state=True),
            dump_folder="",
        )
    assert returned is model and model.lm_head is head
    assert tuple(model.parameters()) == original_parameters
    assert {id(p): p.requires_grad for p in model.parameters()} == requires_grad
    assert len(original_parameters) == len({id(p) for p in original_parameters})
    if skip_dp:
        assert sharded == []
    else:
        assert sharded == [
            model.tok_embeddings,
            model.norm,
            head,
            *model.layers.values(),
            model.mtp_layers[0],
            model,
        ]
    model.audit_mtp_trainability()
    model.train()
    audit = FrozenAudit(model)
    optimizer = torch.optim.AdamW(
        model.mtp_layers.parameters(), lr=0.01, weight_decay=0
    )
    tokens, positions = torch.tensor([1, 2, 3, 4]), torch.arange(4)
    main, auxiliary = model(
        tokens, positions=positions, mtp_input=tokens, mtp_positions=positions
    )
    forward_calls = list(calls)
    assert mtp.calls == 1 and not main.requires_grad and auxiliary.requires_grad
    F.cross_entropy(head(auxiliary), torch.tensor([3, 4, -100, -100])).backward()
    assert mtp.calls == 2 and calls == forward_calls
    assert all(p.grad is not None for p in model.mtp_layers.parameters())
    optimizer.step()
    audit.check(model, [optimizer])
