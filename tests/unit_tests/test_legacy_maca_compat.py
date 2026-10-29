# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.vision_encoder import create_block_diagonal_mask


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_full_recompute_preserves_rng_output_and_gradient(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.completed = 0

            def forward(self, x):
                y = torch.nn.functional.dropout(x.sin(), p=0.3, training=True).square()
                self.completed += 1
                return y

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            x = torch.randn(17, 23, requires_grad=True)
            other = x.detach().clone().requires_grad_()
            reference = Block()
            checkpointed = Block()
            wrapped = FullAC(FullAC.Config())._wrap_block(checkpointed)
            torch.manual_seed(42)
            expected = reference(x)
            expected.sum().backward()
            torch.manual_seed(42)
            actual = wrapped(other)
            actual.sum().backward()
        self.assertEqual(checkpointed.completed, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(other.grad, x.grad, rtol=0, atol=0)


@unittest.skipUnless(
    bool(getattr(torch.version, "maca", None)) and torch.cuda.is_available(),
    "Requires a MACA device",
)
class MacaVisionAttentionTest(unittest.TestCase):
    def test_real_image_size_matches_fp32_forward_and_backward(self):
        # T = patch tokens, H = vision heads, D = head width.
        # This size exposes the MACA max-autotune failure missed by tiny inputs.
        torch.manual_seed(42)
        length = 3520
        q_THD, k_THD, v_THD = [
            torch.randn(
                length, 16, 72, device="cuda", dtype=torch.bfloat16
            ).requires_grad_()
            for _ in range(3)
        ]
        mask = create_block_diagonal_mask(
            torch.tensor([length], device="cuda"), length, torch.device("cuda")
        )
        attention = FlexInnerAttention.Config().build()
        actual_THD = attention(q_THD, k_THD, v_THD, attention_masks=mask)
        qr_THD, kr_THD, vr_THD = [
            value.detach().float().requires_grad_() for value in (q_THD, k_THD, v_THD)
        ]
        scores_HTT = (qr_THD.transpose(0, 1) @ kr_THD.permute(1, 2, 0)) * 72**-0.5
        expected_THD = (scores_HTT.softmax(-1) @ vr_THD.transpose(0, 1)).transpose(0, 1)
        gradient_THD = torch.randn_like(actual_THD)
        actual_THD.backward(gradient_THD)
        expected_THD.backward(gradient_THD.float())

        def relative_error(actual, expected):
            self.assertTrue(torch.isfinite(actual).all())
            return (
                (actual.detach().float() - expected).norm() / expected.norm()
            ).item()

        self.assertLess(relative_error(actual_THD, expected_THD), 0.006)
        for actual, expected in zip(
            (q_THD, k_THD, v_THD), (qr_THD, kr_THD, vr_THD), strict=True
        ):
            self.assertLess(relative_error(actual.grad, expected.grad), 0.015)


@unittest.skipUnless(getattr(torch.version, "maca", None), "MACA GPU integration test")
class PackedConvolutionTests(unittest.TestCase):
    def setUp(self):
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.deterministic, warn_only=self.warn_only)

    def test_forward_backward_against_cpu_float64(self):
        from torchtitan.models.common.causal_conv import packed_causal_conv1d

        generator = torch.Generator().manual_seed(42)
        boundaries = [0, 1, 65, 129]
        for dtype in [torch.bfloat16, torch.float32]:
            for width in [2, 3, 4]:
                for activation in [None, "silu"]:
                    with self.subTest(dtype=dtype, width=width, activation=activation):
                        original = torch.randn(1, 16, 129, generator=generator).to(
                            dtype
                        )
                        weight = torch.randn(16, width, generator=generator).to(dtype)
                        upstream = torch.randn(original.shape, generator=generator).to(
                            dtype
                        )
                        x64 = original.double().requires_grad_(True)
                        w64 = weight.double().requires_grad_(True)
                        spans = []
                        for a, b in zip(boundaries[:-1], boundaries[1:]):
                            y = F.conv1d(
                                x64[:, :, a:b],
                                w64.unsqueeze(1),
                                padding=width - 1,
                                groups=16,
                            )
                            y = y[:, :, : b - a]
                            spans.append(F.silu(y) if activation else y)
                        expected = torch.cat(spans, dim=2)
                        gx, gw = torch.autograd.grad(
                            expected, (x64, w64), upstream.double()
                        )
                        seq = torch.bucketize(
                            torch.arange(129), torch.tensor(boundaries[1:]), right=True
                        )
                        seq = seq.int().unsqueeze(0).cuda()
                        previous = None
                        for _ in range(3):
                            x = (
                                original.transpose(1, 2)
                                .contiguous()
                                .transpose(1, 2)
                                .cuda()
                                .requires_grad_(True)
                            )
                            w = weight.cuda().requires_grad_(True)
                            actual = packed_causal_conv1d(
                                x, w, seq_idx=seq, activation=activation
                            )
                            dx, dw = torch.autograd.grad(
                                actual, (x, w), upstream.cuda()
                            )
                            values = [actual.detach().cpu(), dx.cpu(), dw.cpu()]
                            if previous is not None:
                                self.assertTrue(
                                    all(
                                        torch.equal(a, b)
                                        for a, b in zip(previous, values)
                                    )
                                )
                            previous = values
                            for value, reference in zip(
                                values, [expected.detach(), gx, gw]
                            ):
                                relative = torch.linalg.vector_norm(
                                    value.double() - reference
                                ) / torch.linalg.vector_norm(reference)
                                self.assertLess(
                                    float(relative),
                                    0.006 if dtype == torch.bfloat16 else 2e-5,
                                )


if __name__ == "__main__":
    unittest.main()
