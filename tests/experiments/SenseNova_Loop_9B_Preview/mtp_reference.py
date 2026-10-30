# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Independent tensor reference for packed Qwen MTP, with no model imports.

The norm/fusion contract is cross-checked against archived vLLM source.
Attention uses explicit dense causal matrices for short reference sequences.
T = tokens, D = hidden width, H = query heads, K = head width.
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path


def load_cases():
    return json.loads(
        (Path(__file__).parent / "fixtures" / "mtp_cases.json").read_text()
    )


def align_reference(case):
    """Use explicit document IDs and Python indexing, not reset-position roll."""
    tokens, labels, docs = (case[key] for key in ("input", "labels", "documents"))
    padding = case.get("padding_mask", [False] * len(tokens))
    rope = case.get("mrope_positions", case["positions"])
    rope_zero = [0] * len(rope[0]) if isinstance(rope[0], list) else 0
    result = {
        "mtp_input": [],
        "mtp_labels": [],
        "mtp_input_valid_mask": [],
        "mtp_positions": [],
    }
    if "vision_bank_indices" in case:
        result["mtp_vision_bank_indices"] = []
    for index in range(len(tokens)):
        future = index + 1
        valid = (
            future < len(tokens)
            and docs[index] == docs[future]
            and not (padding[index] or padding[future])
        )
        result["mtp_input"].append(tokens[future] if valid else 0)
        result["mtp_labels"].append(labels[future] if valid else -100)
        result["mtp_input_valid_mask"].append(valid)
        result["mtp_positions"].append(rope[future] if valid else rope_zero)
        if "vision_bank_indices" in case:
            result["mtp_vision_bank_indices"].append(
                case["vision_bank_indices"][future] if valid else -1
            )
    result["mtp_valid_tokens"] = sum(label != -100 for label in result["mtp_labels"])
    return result


def offset_rms_reference(x, weight, eps=1e-6, *, compute_dtype=torch.float32):
    values = x.to(compute_dtype)
    normalized = values / torch.sqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * (1 + weight.to(compute_dtype))).to(x.dtype)


def rotary_reference(
    x_THK, positions, *, rotary_dim, theta, sections, compute_dtype=torch.float32
):
    half = rotary_dim // 2
    frequencies = theta ** (
        -torch.arange(half, device=x_THK.device, dtype=compute_dtype) * 2 / rotary_dim
    )
    if positions.ndim == 1:
        angles = positions.to(compute_dtype)[:, None] * frequencies[None, :]
    else:
        # Interleaved height and width channels; other channels use time.
        coordinates = []
        for channel in range(half):
            axis = channel % 3
            if axis == 0 or channel >= 3 * sections[axis]:
                axis = 0
            coordinates.append(positions[:, axis])
        angles = torch.stack(coordinates, dim=-1).to(compute_dtype) * frequencies
    angles = torch.cat((angles, angles), dim=-1)[:, None, :]
    rotary = x_THK[..., :rotary_dim].to(compute_dtype)
    rotated = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    result = rotary * angles.cos() + rotated * angles.sin()
    return torch.cat((result.to(x_THK.dtype), x_THK[..., rotary_dim:]), dim=-1)


def mtp_reference(
    weights,
    hidden_TD,
    embedding_TD,
    positions,
    documents,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    theta,
    sections,
    eps=1e-6,
    compute_dtype=torch.float32,
    return_intermediates=False,
):
    """Evaluate one layer using source HF keys and primitive tensor operations."""

    def norm(values, key):
        return offset_rms_reference(
            values, weights[key], eps, compute_dtype=compute_dtype
        )

    embedding = norm(embedding_TD, "mtp.pre_fc_norm_embedding.weight")
    hidden = norm(hidden_TD, "mtp.pre_fc_norm_hidden.weight")
    fusion_input = torch.cat((embedding, hidden), dim=-1)
    fused = F.linear(fusion_input, weights["mtp.fc.weight"])
    layer = "mtp.layers.0."
    attn = layer + "self_attn."
    inputs = norm(fused, layer + "input_layernorm.weight")
    query_gate = F.linear(inputs, weights[attn + "q_proj.weight"]).reshape(
        -1, num_heads, 2, head_dim
    )
    query, gate = query_gate.unbind(dim=2)
    key = F.linear(inputs, weights[attn + "k_proj.weight"]).reshape(
        -1, num_kv_heads, head_dim
    )
    value = F.linear(inputs, weights[attn + "v_proj.weight"]).reshape(
        -1, num_kv_heads, head_dim
    )
    query = norm(query, attn + "q_norm.weight")
    key = norm(key, attn + "k_norm.weight")
    kwargs = dict(
        rotary_dim=rotary_dim,
        theta=theta,
        sections=sections,
        compute_dtype=compute_dtype,
    )
    query = rotary_reference(query, positions, **kwargs)
    key = rotary_reference(key, positions, **kwargs)
    rotary_key = key
    key = key.repeat_interleave(num_heads // num_kv_heads, dim=1)
    value = value.repeat_interleave(num_heads // num_kv_heads, dim=1)
    scores = (
        torch.einsum("thk,shk->hts", query.to(compute_dtype), key.to(compute_dtype))
        * head_dim**-0.5
    )
    slots = torch.arange(hidden_TD.shape[0], device=hidden_TD.device)
    visible = (documents[:, None] == documents[None, :]) & (
        slots[:, None] >= slots[None, :]
    )
    probability = scores.masked_fill(~visible, -torch.inf).softmax(dim=-1)
    attended = torch.einsum("hts,shk->thk", probability, value.to(compute_dtype)).to(
        inputs.dtype
    )
    attended = (attended * gate.sigmoid()).reshape(inputs.shape[0], -1)
    residual = fused + F.linear(attended, weights[attn + "o_proj.weight"])
    inputs = norm(residual, layer + "post_attention_layernorm.weight")
    ff = F.silu(F.linear(inputs, weights[layer + "mlp.gate_proj.weight"]))
    ff = ff * F.linear(inputs, weights[layer + "mlp.up_proj.weight"])
    output = residual + F.linear(ff, weights[layer + "mlp.down_proj.weight"])
    result = norm(output, "mtp.norm.weight")
    if return_intermediates:
        return result, {
            "enorm": embedding,
            "hnorm": hidden,
            "fusion_input": fusion_input,
            "fused": fused,
            "rotary_query": query,
            "rotary_key": rotary_key,
            "gated_attention": attended,
        }
    return result


def objective_reference(
    main_logits,
    mtp_logits,
    main_labels,
    mtp_labels,
    *,
    mode,
    coefficient,
    main_count=None,
    mtp_count=None,
):
    """Independent unreduced CE with distinct optimizer-step denominators."""
    main_count = (main_labels != -100).sum() if main_count is None else main_count
    mtp_count = (mtp_labels != -100).sum() if mtp_count is None else mtp_count
    main_sum = F.cross_entropy(main_logits, main_labels, reduction="sum")
    mtp_sum = F.cross_entropy(mtp_logits, mtp_labels, reduction="sum")
    main = main_sum / torch.as_tensor(main_count, device=main_sum.device).clamp_min(1)
    mtp = mtp_sum / torch.as_tensor(mtp_count, device=mtp_sum.device).clamp_min(1)
    total = coefficient * mtp
    if mode == "joint":
        total = total + main
    return total, main.detach(), mtp.detach()
