# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared FP32 test references, independent of production fused kernels."""

import torch


def install_fp32_reference_kernels():
    """Test-only FP32 kernels; production fused GDN only accepts FP16/BF16."""
    from attn_gym.linear import chunk_gdn
    from torchtitan.models.qwen3_5.gdn import GatedDeltaKernel
    from torchtitan.models.common.attention import VarlenInnerAttention

    def delta(self, xq_THK, xk_THK, xv_THV, g_TH, beta_TH, *, cu_seqlens=None):
        q, k, v, g, beta = xq_THK, xk_THK, xv_THV, g_TH, beta_TH
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        output, _ = chunk_gdn(
            q[None],
            k[None],
            v[None],
            g[None],
            beta[None],
            cu_seqlens=cu_seqlens,
            scale=q.shape[-1] ** -0.5,
            impl="reference",
        )
        return output[0]

    def attention(
        self,
        q_THK,
        k_THK,
        v_THV,
        *,
        attention_masks,
        scale=None,
        out_transform=None,
        **kwargs,
    ):
        q, k, v = q_THK, k_THK, v_THV
        q_offsets = attention_masks.cu_seq_q.tolist()
        k_offsets = attention_masks.cu_seq_k.tolist()
        outputs = []
        for qa, qb, ka, kb in zip(
            q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:]
        ):
            query, key, value = (
                q[qa:qb].transpose(0, 1),
                k[ka:kb].transpose(0, 1),
                v[ka:kb].transpose(0, 1),
            )
            key = key.repeat_interleave(query.shape[0] // key.shape[0], dim=0)
            value = value.repeat_interleave(query.shape[0] // value.shape[0], dim=0)
            logits = (query @ key.transpose(-1, -2)) * (scale or q.shape[-1] ** -0.5)
            qpos = (
                torch.arange(qb - qa, device=q.device)[:, None] + (kb - ka) - (qb - qa)
            )
            kpos = torch.arange(kb - ka, device=q.device)[None, :]
            if self.window_size[1] == 0:
                logits = logits.masked_fill(kpos > qpos, -torch.inf)
            output = logits.softmax(-1) @ value
            assert out_transform is None
            outputs.append(output.transpose(0, 1))
        return torch.cat(outputs)

    GatedDeltaKernel.forward = delta
    VarlenInnerAttention.forward = attention
