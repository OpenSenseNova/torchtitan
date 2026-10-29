# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical checks independent of model assets and GPUs."""

import copy

import pytest
import torch
import torch.nn.functional as F

from torchtitan.experiments.loop_qwen35.dpo.loss import dpo_loss
from torchtitan.components.sequence_logps import (
    sequence_logps,
    weighted_sequence_backward,
)


@pytest.mark.parametrize("beta", [0.01, 0.1, 2.0])
def test_sigmoid_loss_and_coefficients(beta):
    torch.manual_seed(4)
    scores = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    ref = torch.randn_like(scores)
    valid = torch.tensor([True, True, False, True])
    loss, coeff, metrics = dpo_loss(scores, ref, valid, beta=beta, global_pairs=3)
    delta = (scores[:, 0] - scores[:, 1]) - (ref[:, 0] - ref[:, 1])
    expected = (-F.logsigmoid(beta * delta))[valid].mean()
    torch.testing.assert_close(loss, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        torch.autograd.grad(expected, scores)[0], coeff, atol=1e-10, rtol=1e-10
    )
    assert metrics["valid_pairs"] == 3
    swapped, swapped_coeff, _ = dpo_loss(
        scores.flip(-1), ref.flip(-1), valid, beta=beta, global_pairs=3
    )
    swapped_expected = (-F.logsigmoid(-beta * delta))[valid].mean()
    torch.testing.assert_close(swapped, swapped_expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        torch.autograd.grad(swapped_expected, scores)[0],
        swapped_coeff.flip(-1),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("chunks", [1, 2, 4, 8])
@pytest.mark.parametrize("tied", [False, True])
def test_two_pass_loss_and_gradient(chunks, tied):
    torch.manual_seed(7)
    embedding = torch.nn.Embedding(11, 6, dtype=torch.float64)
    head = torch.nn.Linear(6, 11, bias=False, dtype=torch.float64)
    if tied:
        head.weight = embedding.weight
    copy_embedding, copy_head = copy.deepcopy((embedding, head))
    inputs = torch.randint(0, 11, (16,))
    labels = torch.randint(0, 11, (16,))
    labels[[0, 3, 8, 14, 15]] = -100
    ids = torch.tensor([0] * 3 + [1] * 5 + [2] * 2 + [3] * 4 + [-1] * 2)
    valid = torch.tensor([True, True])
    reference = torch.randn(2, 2, dtype=torch.float64)
    full = sequence_logps(embedding(inputs).sin(), head, labels, ids, 4).reshape(2, 2)
    loss, _, _ = dpo_loss(full, reference, valid)
    loss.backward()
    hidden = copy_embedding(inputs).sin()
    with torch.no_grad():
        scores = sequence_logps(
            hidden.detach(), copy_head, labels, ids, 4, num_chunks=chunks
        ).reshape(2, 2)
        chunk_loss, coefficients, _ = dpo_loss(scores, reference, valid)
    weighted_sequence_backward(
        hidden, copy_head, labels, ids, coefficients.flatten(), num_chunks=chunks
    )
    for actual, expected in (
        (scores, full),
        (chunk_loss, loss),
        (copy_embedding.weight.grad, embedding.weight.grad),
        (copy_head.weight.grad, head.weight.grad),
    ):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_initial_reference_and_padding():
    scores = torch.tensor([[-2.0, -4.0], [0.0, 0.0]], requires_grad=True)
    loss, coeff, metrics = dpo_loss(
        scores, scores.detach(), torch.tensor([True, False])
    )
    torch.testing.assert_close(loss, torch.tensor(2.0).log())
    assert metrics["preference_accuracy"] == 0
    assert torch.count_nonzero(coeff[1]) == 0


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf")])
def test_invalid_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.ones(1, dtype=torch.bool),
            beta=beta,
        )
