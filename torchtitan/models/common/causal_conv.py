# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Packed causal convolution with a deterministic FP32 weight reduction.

The native forward and input gradient are preserved. The vendor weight gradient
uses atomic accumulation and can vary even with deterministic algorithms enabled.
"""
import causal_conv1d_cuda
import torch


def _shifted_input(x_BCT, sequence_ids_BT, offset):
    if offset == 0:
        return x_BCT.float()
    shifted_BCT = torch.zeros_like(x_BCT, dtype=torch.float32)
    valid_BT = sequence_ids_BT[:, offset:] == sequence_ids_BT[:, :-offset]
    shifted_BCT[:, :, offset:] = x_BCT[:, :, :-offset].float() * valid_BT[:, None, :]
    return shifted_BCT


class _DeterministicPackedCausalConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_BCT, weight_CW, sequence_ids_BT, activation):
        if activation not in [None, "silu", "swish"]:
            raise ValueError("Unsupported activation")
        ctx.activation = activation in ["silu", "swish"]
        ctx.save_for_backward(x_BCT, weight_CW, sequence_ids_BT)
        return causal_conv1d_cuda.causal_conv1d_fwd(
            x_BCT, weight_CW, None, sequence_ids_BT, None, None, ctx.activation
        )

    @staticmethod
    def backward(ctx, upstream_BCT):
        x_BCT, weight_CW, sequence_ids_BT = ctx.saved_tensors
        if upstream_BCT.stride(2) != 1 and upstream_BCT.stride(1) != 1:
            upstream_BCT = upstream_BCT.contiguous()
        dx_BCT, _, _, _ = causal_conv1d_cuda.causal_conv1d_bwd(
            x_BCT,
            weight_CW,
            None,
            upstream_BCT,
            sequence_ids_BT,
            None,
            None,
            None,
            False,
            ctx.activation,
        )
        width = weight_CW.shape[1]
        preactivation_BCT = torch.zeros_like(x_BCT, dtype=torch.float32)
        for index in range(width):
            shifted_BCT = _shifted_input(x_BCT, sequence_ids_BT, width - 1 - index)
            preactivation_BCT = torch.addcmul(
                preactivation_BCT,
                shifted_BCT,
                weight_CW[:, index].float()[None, :, None],
            )
        derivative_BCT = upstream_BCT.float()
        if ctx.activation:
            sigmoid_BCT = torch.sigmoid(preactivation_BCT)
            derivative_BCT = (
                derivative_BCT
                * sigmoid_BCT
                * (1 + preactivation_BCT * (1 - sigmoid_BCT))
            )
        gradient_columns_C = []
        for index in range(width):
            shifted_BCT = _shifted_input(x_BCT, sequence_ids_BT, width - 1 - index)
            gradient_columns_C.append((shifted_BCT * derivative_BCT).sum(dim=(0, 2)))
        dw_CW = torch.stack(gradient_columns_C, dim=1).to(weight_CW.dtype)
        return dx_BCT, dw_CW, None, None


def packed_causal_conv1d(x_BCT, weight_CW, *, seq_idx, activation):
    """Keep the native seq_idx keyword for interchangeable backend dispatch."""
    return _DeterministicPackedCausalConv.apply(x_BCT, weight_CW, seq_idx, activation)
