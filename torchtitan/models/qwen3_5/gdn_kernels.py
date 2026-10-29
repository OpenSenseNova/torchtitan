# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GDN kernel contracts for Attention Gym and the MACA FLA implementation."""
import torch

if not getattr(torch.version, "maca", None):
    from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm, recurrent_gdn
else:
    import triton
    from causal_conv1d import causal_conv1d_fn
    from fla.modules.l2norm import l2norm as vendor_l2norm
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    )
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    from torchtitan.models.common.causal_conv import packed_causal_conv1d

    # The vendor's 2-4 stage candidates exceed C500's 64 KiB shared memory.
    _backward_autotuner = chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64.fn
    _backward_autotuner.configs = [triton.Config({"BV": 32}, num_warps=4, num_stages=1)]
    _backward_autotuner.cache.clear()

    def _expand_key_heads(q, k, v):
        if q.shape[2] != k.shape[2] or v.shape[2] % q.shape[2]:
            raise ValueError(
                "GDN value heads must be a multiple of matching query/key heads"
            )
        repeats = v.shape[2] // q.shape[2]
        if repeats > 1:
            q = q.repeat_interleave(repeats, dim=2)
            k = k.repeat_interleave(repeats, dim=2)
        return q, k

    def l2norm(x, *, cu_seqlens=None):
        # Normalization is independent for every token and head.
        return vendor_l2norm(x, eps=1e-6)

    def chunk_gdn(q, k, v, g, beta, *, cu_seqlens=None, scale=None, impl="fused"):
        if impl != "fused":
            raise ValueError("The MACA adapter supports the fused GDN contract only")
        q, k = _expand_key_heads(q, k, v)
        return chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )

    def causal_conv1d(x, weight, *, activation, cu_seqlens):
        if x.shape[0] != 1:
            raise ValueError("Packed convolution requires a single flattened batch")
        positions = torch.arange(x.shape[1], device=x.device)
        seq_idx = (
            torch.bucketize(positions, cu_seqlens[1:], right=True)
            .to(torch.int32)
            .unsqueeze(0)
        )
        conv_fn = (
            packed_causal_conv1d
            if torch.are_deterministic_algorithms_enabled()
            else causal_conv1d_fn
        )
        output = conv_fn(
            x.transpose(1, 2),
            weight,
            activation=activation,
            seq_idx=seq_idx,
        )
        return output.transpose(1, 2)

    def recurrent_gdn(
        q,
        k,
        v,
        g,
        beta,
        state_cache,
        *,
        cu_seqlens,
        scale,
        state_indices,
        has_initial_state,
        autotune=False
    ):
        q, k = _expand_key_heads(q, k, v)
        initial = (
            state_cache.index_select(0, state_indices.long())
            .transpose(-1, -2)
            .contiguous()
        )
        initial = torch.where(
            has_initial_state[:, None, None, None], initial, torch.zeros_like(initial)
        )
        output, final = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )
        state_cache.index_copy_(0, state_indices.long(), final.transpose(-1, -2))
        return output, state_cache
