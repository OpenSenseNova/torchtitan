# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in N4 merge, preserving BF16 round points with optional activation-gradient fusion."""

import torch
import triton
import triton.language as tl


@triton.jit
def _merge4_forward(X, Branch, Post, Mix, Out, D: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = channel < D
    base = token * 4 * D + channel
    x0 = tl.load(X + base, valid, other=0).to(tl.float32)
    x1 = tl.load(X + base + D, valid, other=0).to(tl.float32)
    x2 = tl.load(X + base + 2 * D, valid, other=0).to(tl.float32)
    x3 = tl.load(X + base + 3 * D, valid, other=0).to(tl.float32)
    branch = tl.load(Branch + token * D + channel, valid, other=0).to(tl.float32)
    for stream in tl.static_range(4):
        m0 = tl.load(Mix + token * 16 + stream * 4).to(tl.float32)
        m1 = tl.load(Mix + token * 16 + stream * 4 + 1).to(tl.float32)
        m2 = tl.load(Mix + token * 16 + stream * 4 + 2).to(tl.float32)
        m3 = tl.load(Mix + token * 16 + stream * 4 + 3).to(tl.float32)
        mixed = ((m0 * x0 + m1 * x1) + m2 * x2) + m3 * x3
        # The eager bmm and multiply each produce BF16 before the final add.
        mixed = mixed.to(tl.bfloat16).to(tl.float32)
        post = tl.load(Post + token * 4 + stream).to(tl.float32)
        update = (post * branch).to(tl.bfloat16).to(tl.float32)
        tl.store(Out + base + stream * D, mixed + update, valid)


class _FusedMerge4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, branch, post, mix, fused_backward):
        ctx.fused_backward = fused_backward
        ctx.save_for_backward(residual, branch, post, mix)
        output = torch.empty_like(residual)
        tokens, _, dim = residual.shape
        block = min(triton.next_power_of_2(dim), 1024)
        if tokens:
            with torch.cuda.device(residual.device):
                _merge4_forward[(tokens, triton.cdiv(dim, block))](
                    residual,
                    branch,
                    post,
                    mix,
                    output,
                    dim,
                    block,
                    num_warps=8 if block >= 512 else 4,
                    enable_fp_fusion=False,
                )
        return output

    @staticmethod
    def backward(ctx, gradient):
        residual, branch, post, mix = ctx.saved_tensors
        need_x, need_b, need_p, need_m, _ = ctx.needs_input_grad
        if ctx.fused_backward:
            from torchtitan.experiments.SenseNova_Loop_9B_Preview.mhc.fused_backward import (
                merge_activation_gradients,
            )

            grad_x, grad_b = merge_activation_gradients(
                gradient, post, mix, need_x, need_b
            )
        else:
            grad_x = torch.matmul(mix.transpose(-2, -1), gradient) if need_x else None
            grad_b = (gradient * post.unsqueeze(-1)).sum(-2) if need_b else None
        grad_p = (gradient * branch.unsqueeze(-2)).sum(-1) if need_p else None
        grad_m = torch.matmul(gradient, residual.transpose(-2, -1)) if need_m else None
        return grad_x, grad_b, grad_p, grad_m, None


def fused_merge4(residual, branch, post, mix, *, fused_backward=False):
    if residual.device.type != "cuda" or residual.dtype != torch.bfloat16:
        raise ValueError("fused_merge4 requires CUDA BF16 residuals")
    if residual.ndim != 3 or residual.shape[1] != 4:
        raise ValueError("fused_merge4 requires [tokens, 4, dim]")
    # The kernel uses signed 32-bit element offsets.
    if residual.shape[2] == 0 or residual.numel() > 2**31:
        raise ValueError(
            "fused_merge4 requires dim > 0 and at most 2**31 residual elements"
        )
    if any(value.device != residual.device for value in (branch, post, mix)):
        raise ValueError("fused_merge4 requires all inputs on the same CUDA device")
    if (
        branch.shape != (residual.shape[0], residual.shape[2])
        or branch.dtype != residual.dtype
    ):
        raise ValueError("branch must be BF16 [tokens, dim]")
    if post.shape != residual.shape[:2] or mix.shape != (*residual.shape[:2], 4):
        raise ValueError("post/mix shapes do not match four streams")
    if not residual.is_contiguous() or not branch.is_contiguous():
        raise ValueError("fused_merge4 requires contiguous residual and branch")
    return _FusedMerge4.apply(
        residual,
        branch,
        post.to(residual.dtype).contiguous(),
        mix.to(residual.dtype).contiguous(),
        fused_backward,
    )
