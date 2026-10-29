# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch

from torchtitan.experiments.loop_qwen35.dpo.reference import (
    copy_local_state,
    preserve_rng,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear


def test_reference_copy_runs_fused_ffn_load_hooks():
    config = FeedForward.Config(
        w13=Linear.Config(in_features=4, out_features=16),
        w2=Linear.Config(in_features=8, out_features=4),
    )
    policy, reference = config.build(), config.build()
    reference.requires_grad_(False)
    copy_local_state(policy, reference)
    for a, b in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert a.data_ptr() != b.data_ptr()
    original = [p.detach().clone() for p in reference.parameters()]
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(1.0)
    for a, b in zip(original, reference.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_reference_preserves_random_streams():
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    with preserve_rng():
        expected = (random.random(), np.random.random(), torch.rand(3))
    with preserve_rng():
        random.random()
        np.random.random(5)
        torch.rand(12)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)
