# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Local residual routing. T=tokens, N=streams, D=hidden channels."""

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.qwen3_5.model import Qwen35TransformerBlock
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig


def sinkhorn(logits, iterations=20):
    # Log-domain iterations avoid overflow without changing the positive map.
    z = logits.float()
    for _ in range(iterations):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
    return z.exp()


def expand_streams(x_TD, streams):
    return (
        x_TD.unsqueeze(-2)
        .expand(*x_TD.shape[:-1], streams, x_TD.shape[-1])
        .contiguous()
        .flatten(-2)
    )


def collapse_streams(x_TND, streams):
    return x_TND.unflatten(-1, (streams, -1)).mean(dim=-2)


class MHCRouter(Module):
    """Dynamic mHC router with near function-preserving initialization.

    Nonuniform initial pre weights sum to one, while post weights equal one.
    Equal input streams therefore preserve the original function at init.
    Unequal pre weights break backward symmetry between the output streams.
    """

    def __init__(
        self,
        dim,
        streams=2,
        iterations=20,
        fixed_identity=False,
        pre_scale=1.0,
        fused_backward=False,
        fused_normalization=False,
    ):
        super().__init__()
        self.dim, self.streams, self.iterations = dim, streams, iterations
        self.fixed_identity = fixed_identity
        self.pre_scale = pre_scale
        self.fused_backward = fused_backward
        self.fused_normalization = fused_normalization
        self.collect_diagnostics = True
        count = 2 * streams + (0 if fixed_identity else streams * streams)
        self.weight = nn.Parameter(
            torch.empty(count, streams * dim, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(count, dtype=torch.float32))
        self._sharding_config = ShardingConfig(
            state_shardings={
                name: dense_param_placement(tp=spmd.I) for name in ("weight", "bias")
            }
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.weight.zero_()
        n = self.streams
        b = torch.zeros(self.bias.shape, dtype=torch.float32, device=self.bias.device)
        # Dyadic coefficients remain exactly representable in BF16.
        values = {2: (0.25, 0.75), 4: (0.0625, 0.1875, 0.25, 0.5)}
        pre = torch.tensor(values[n], dtype=torch.float32, device=b.device)
        b[:n] = torch.logit(pre)
        if not self.fixed_identity:
            diagonal_logit = math.log((1 - (n - 1) / 16) * 16)
            b[2 * n :] = (torch.eye(n, device=b.device) * diagonal_logit).flatten()
        if isinstance(self.bias, DTensor):
            b = distribute_tensor(b, self.bias.device_mesh, self.bias.placements)
        self.bias.copy_(b)

    def forward(self, residual_TND):
        flat = residual_TND.flatten(-2)
        if self.fused_normalization:
            from torchtitan.experiments.loop_qwen35.mhc.fused_norm import router_rmsnorm

            normalized = router_rmsnorm(flat)
        else:
            flat = flat.float()
            normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        logits = torch.nn.functional.linear(
            normalized, self.weight.float(), self.bias.float()
        )
        n = self.streams
        pre = logits[..., :n].sigmoid()
        post = 2 * logits[..., n : 2 * n].sigmoid()
        if self.fixed_identity:
            mix = torch.eye(n, device=flat.device, dtype=torch.float32).expand(
                *flat.shape[:-1], n, n
            )
        else:
            mix = sinkhorn(logits[..., 2 * n :].unflatten(-1, (n, n)), self.iterations)
        if self.training and self.collect_diagnostics:
            with torch.no_grad():
                sample = residual_TND[:64].float()
                self.last_diagnostics = torch.stack(
                    (
                        (mix[:64].sum(-1) - 1).abs().max(),
                        (mix[:64].sum(-2) - 1).abs().max(),
                        sample.var(dim=-2, unbiased=False).mean(),
                    )
                ).detach()
        # Main residual stays in the backbone dtype; matrix operations accumulate
        # using the backend's matmul policy. Small routing computations use FP32.
        if self.fused_backward:
            from torchtitan.experiments.loop_qwen35.mhc.fused_backward import (
                router_input,
            )

            branch_TD = router_input(residual_TND, pre.to(residual_TND.dtype))
        else:
            branch_TD = torch.matmul(
                pre.to(residual_TND.dtype).unsqueeze(-2), residual_TND
            ).squeeze(-2)
        if self.pre_scale != 1.0:
            branch_TD = branch_TD * self.pre_scale
        return branch_TD, post, mix

    @staticmethod
    def merge(
        residual_TND, branch_TD, post_TN, mix_TNN, *, fused=False, fused_backward=False
    ):
        if fused:
            from torchtitan.experiments.loop_qwen35.mhc.fused_merge import fused_merge4

            return fused_merge4(
                residual_TND, branch_TD, post_TN, mix_TNN, fused_backward=fused_backward
            )
        return torch.matmul(mix_TNN.to(residual_TND.dtype), residual_TND) + post_TN.to(
            branch_TD.dtype
        ).unsqueeze(-1) * branch_TD.unsqueeze(-2)


class MHCBlock(Qwen35TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35TransformerBlock.Config):
        mhc_dim: int
        mhc_streams: int = 2
        mhc_iterations: int = 20
        mhc_fixed_identity: bool = False
        mhc_pre_scale: float = 1.0
        mhc_fused_merge: bool = False
        mhc_fused_backward: bool = False
        mhc_fused_normalization: bool = False

    def __init__(self, config):
        super().__init__(config)
        self.mhc_streams = config.mhc_streams
        self.mhc_fused_merge = config.mhc_fused_merge
        self.mhc_fused_backward = config.mhc_fused_backward
        args = (
            config.mhc_dim,
            config.mhc_streams,
            config.mhc_iterations,
            config.mhc_fixed_identity,
            config.mhc_pre_scale,
            config.mhc_fused_backward,
            config.mhc_fused_normalization,
        )
        self.mhc_attn = MHCRouter(*args)
        self.mhc_ffn = MHCRouter(*args)

    def forward(self, x_TD, attention_masks, positions=None, *, padding_mask=None):
        residual = x_TD.unflatten(-1, (self.mhc_streams, -1))
        branch, post, mix = self.mhc_attn(residual)
        branch = self.attention_norm(branch)
        mask = (
            attention_masks[self.attn_mask_key] if attention_masks is not None else None
        )
        branch = (
            self.attn(branch, mask, positions)
            if self.full_attn
            else self.attn(branch, mask)
        )
        residual = self.mhc_attn.merge(
            residual,
            branch,
            post,
            mix,
            fused=self.mhc_fused_merge,
            fused_backward=self.mhc_fused_backward,
        )
        branch, post, mix = self.mhc_ffn(residual)
        branch = self.ffn_norm(branch)
        branch = (
            self.moe(branch, padding_mask_T=padding_mask)
            if self.moe_enabled
            else self.feed_forward(branch)
        )
        return self.mhc_ffn.merge(
            residual,
            branch,
            post,
            mix,
            fused=self.mhc_fused_merge,
            fused_backward=self.mhc_fused_backward,
        ).flatten(-2)
