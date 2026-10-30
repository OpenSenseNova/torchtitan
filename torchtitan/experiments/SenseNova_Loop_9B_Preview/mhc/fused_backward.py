# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve N4 gradient reductions while avoiding a K=1 batched matrix product."""

import torch
import triton
import triton.language as tl


class _RouterInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual_TND, pre_TN):
        ctx.save_for_backward(residual_TND, pre_TN)
        return torch.matmul(pre_TN.unsqueeze(-2), residual_TND).squeeze(-2)

    @staticmethod
    def backward(ctx, gradient_TD):
        residual_TND, pre_TN = ctx.saved_tensors
        need_x, need_pre = ctx.needs_input_grad
        # K=1 bmm is an outer product: no reduction or rounding point changes.
        grad_x = pre_TN.unsqueeze(-1) * gradient_TD.unsqueeze(-2) if need_x else None
        grad_pre = (
            torch.matmul(
                gradient_TD.unsqueeze(-2), residual_TND.transpose(-2, -1)
            ).squeeze(-2)
            if need_pre
            else None
        )
        return grad_x, grad_pre


def router_input(residual_TND, pre_TN):
    return _RouterInput.apply(residual_TND, pre_TN)


@triton.jit
def _merge4_backward_branch(
    Grad,
    Post,
    GradBranch,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = channel < D
    base = token * 4 * D + channel
    g0 = tl.load(Grad + base, valid, other=0).to(tl.float32)
    g1 = tl.load(Grad + base + D, valid, other=0).to(tl.float32)
    g2 = tl.load(Grad + base + 2 * D, valid, other=0).to(tl.float32)
    g3 = tl.load(Grad + base + 3 * D, valid, other=0).to(tl.float32)
    p0 = tl.load(Post + token * 4).to(tl.float32)
    p1 = tl.load(Post + token * 4 + 1).to(tl.float32)
    p2 = tl.load(Post + token * 4 + 2).to(tl.float32)
    p3 = tl.load(Post + token * 4 + 3).to(tl.float32)
    # Eager multiplication materializes BF16 before its FP32 reduction.
    b0 = (p0 * g0).to(tl.bfloat16).to(tl.float32)
    b1 = (p1 * g1).to(tl.bfloat16).to(tl.float32)
    b2 = (p2 * g2).to(tl.bfloat16).to(tl.float32)
    b3 = (p3 * g3).to(tl.bfloat16).to(tl.float32)
    tl.store(GradBranch + token * D + channel, ((b0 + b1) + b2) + b3, valid)


def merge_activation_gradients(gradient_TND, post_TN, mix_TNN, need_x, need_b):
    tokens, _, dim = gradient_TND.shape
    # Preserve cuBLAS reduction order for the K=4 residual gradient.
    grad_x = torch.matmul(mix_TNN.transpose(-2, -1), gradient_TND) if need_x else None
    grad_b = gradient_TND.new_empty((tokens, dim)) if need_b else None
    if tokens and need_b:
        gradient_TND = gradient_TND.contiguous()
        block = min(triton.next_power_of_2(dim), 1024)
        with torch.cuda.device(gradient_TND.device):
            _merge4_backward_branch[(tokens, triton.cdiv(dim, block))](
                gradient_TND,
                post_TN,
                grad_b,
                dim,
                block,
                num_warps=8 if block >= 512 else 4,
                enable_fp_fusion=False,
            )
    return grad_x, grad_b
