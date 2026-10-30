# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse router casts and pointwise work while preserving native FP32 reductions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _square_cast(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(Y + offsets, x * x, offsets < N)


@triton.jit
def _normalize(X, R, Y, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    r = tl.load(R + offsets // WIDTH, offsets < N, other=0)
    tl.store(Y + offsets, x * r, offsets < N)


@triton.jit
def _product(X, DY, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    dy = tl.load(DY + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(Y + offsets, dy * x, offsets < N)


@triton.jit
def _backward(
    X, DY, R, DS, DX, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    dy = tl.load(DY + offsets, offsets < N, other=0).to(tl.float32)
    r = tl.load(R + offsets // WIDTH, offsets < N, other=0)
    ds = tl.load(DS + offsets // WIDTH, offsets < N, other=0)
    # Preserve the rounding of each eager FP32 operation, including the final add.
    dx = tl.inline_asm_elementwise(
        "{ .reg .f32 a, b, c; mul.rn.f32 a, $1, $2; mul.rn.f32 b, $3, 0f40000000; "
        "mul.rn.f32 c, b, $4; add.rn.f32 $0, a, c; }",
        constraints="=f,f,f,f,f",
        args=[dy, r, x, ds],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(DX + offsets, dx, offsets < N)


class _RouterRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        squares = torch.empty_like(x, dtype=torch.float32)
        grid = (triton.cdiv(x.numel(), 1024),)
        if x.numel():
            with torch.cuda.device(x.device):
                _square_cast[grid](x, squares, x.numel(), 1024, enable_fp_fusion=False)
        r = torch.rsqrt(squares.mean(-1, keepdim=True) + 1e-6)
        del squares
        output = torch.empty_like(x, dtype=torch.float32)
        if x.numel():
            with torch.cuda.device(x.device):
                _normalize[grid](
                    x, r, output, x.numel(), x.shape[-1], 1024, enable_fp_fusion=False
                )
        ctx.save_for_backward(x, r)
        return output

    @staticmethod
    def backward(ctx, dy):
        x, r = ctx.saved_tensors
        dy = dy.contiguous()
        product = torch.empty_like(x, dtype=torch.float32)
        grid = (triton.cdiv(x.numel(), 1024),)
        if x.numel():
            with torch.cuda.device(x.device):
                _product[grid](x, dy, product, x.numel(), 1024, enable_fp_fusion=False)
        dr = product.sum(-1, keepdim=True)
        del product
        ds = (dr * (-0.5 * r.pow(3))) / x.shape[-1]
        dx = torch.empty_like(x)
        if x.numel():
            with torch.cuda.device(x.device):
                _backward[grid](
                    x,
                    dy,
                    r,
                    ds,
                    dx,
                    x.numel(),
                    x.shape[-1],
                    1024,
                    enable_fp_fusion=False,
                )
        return dx


def router_rmsnorm(x):
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Router normalization requires a CUDA BF16 or FP32 tensor")
    if x.ndim != 2 or not x.is_contiguous() or x.shape[-1] <= 0 or x.numel() >= 2**31:
        raise ValueError(
            "Router normalization requires a contiguous 2D tensor with 32-bit indexing"
        )
    return _RouterRMSNorm.apply(x)
