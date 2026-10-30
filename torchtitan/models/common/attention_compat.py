# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Attention APIs shared by upstream PyTorch and vendor PyTorch 2.8 builds."""
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch.nn.attention.flex_attention import flex_attention as _torch_flex_attention

try:
    from torch.nn.attention import (
        activate_flash_attention_impl,
        current_flash_attention_impl,
        restore_flash_attention_impl,
    )
    from torch.nn.attention.varlen import AuxRequest as VarlenAuxRequest, varlen_attn
except ImportError:

    @dataclass(frozen=True)
    class VarlenAuxRequest:
        lse: bool = False

    def current_flash_attention_impl():
        return None

    def activate_flash_attention_impl(implementation):
        if implementation != "FA2":
            raise ValueError(
                "This PyTorch build supports the external FA2 backend only"
            )

    def restore_flash_attention_impl():
        # External FA2 has no process-wide dispatch registration to restore.
        return None

    def varlen_attn(
        q,
        k,
        v,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        *,
        scale=None,
        window_size=(-1, 0),
        return_aux=None
    ):
        """Use the vendor FA2 autograd implementation for packed attention."""
        from flash_attn import flash_attn_varlen_func

        need_lse = return_aux is not None and return_aux.lse
        result = flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=window_size[1] == 0,
            window_size=window_size,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            return_attn_probs=need_lse,
        )
        if need_lse:
            output, lse, _ = result
            return output, lse
        return result


try:
    from torch.nn.attention.flex_attention import AuxRequest

    flex_attention = _torch_flex_attention
except ImportError:

    @dataclass(frozen=True)
    class AuxRequest:
        lse: bool = False
        max_scores: bool = False

    class _FlexAuxOutput(NamedTuple):
        lse: torch.Tensor | None
        max_scores: torch.Tensor | None = None

    def flex_attention(
        q,
        k,
        v,
        *,
        score_mod=None,
        block_mask=None,
        scale=None,
        enable_gqa=False,
        return_aux=None,
        kernel_options=None
    ):
        if return_aux is not None and return_aux.max_scores:
            raise NotImplementedError("Flex max_scores requires a newer PyTorch build")
        need_lse = return_aux is not None and return_aux.lse
        query_width, value_width = q.shape[-1], v.shape[-1]
        # The vendor Triton 3.0 Flex kernel requires power-of-two head widths.
        # Zero padding preserves scores when the original scale is retained.
        if scale is None:
            scale = query_width**-0.5
        query_pad = (1 << (query_width - 1).bit_length()) - query_width
        value_pad = (1 << (value_width - 1).bit_length()) - value_width
        if query_pad:
            q = torch.nn.functional.pad(q, (0, query_pad))
            k = torch.nn.functional.pad(k, (0, query_pad))
        if value_pad:
            v = torch.nn.functional.pad(v, (0, value_pad))
        if getattr(torch.version, "maca", None):
            # MACA 3.5.3 rejects this kernel's multi-stage load dependencies.
            kernel_options = dict(kernel_options or {})
            kernel_options.setdefault("num_stages", 1)
            # Default backward tiles exceed C500's 64 KiB shared memory.
            for option in ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"):
                kernel_options.setdefault(option, 32)
        result = _torch_flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=need_lse,
            kernel_options=kernel_options,
        )
        if need_lse:
            output, lse = result
            return output[..., :value_width], _FlexAuxOutput(lse)
        output = result[..., :value_width]
        if return_aux is None:
            return output
        return output, _FlexAuxOutput(None)


def varlen_attention_op():
    """Return the opaque op required by attention-selective checkpointing."""
    if hasattr(torch.ops.torch_attn, "_varlen_attn"):
        return torch.ops.torch_attn._varlen_attn.default
    raise ValueError(
        "Attention-selective checkpointing requires PyTorch's varlen custom op; "
        "use the full checkpoint policy with the external FA2 backend"
    )
