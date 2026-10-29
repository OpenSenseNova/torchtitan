# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Compare MTP objective and gradients with independent CE mathematics."""

import pytest
import torch
from torchtitan.components.loss import ChunkedLossWrapper

from torchtitan.experiments.loop_qwen35.loss import LoopMTPLoss, normalize_mtp_losses
from mtp_reference import objective_reference


@pytest.mark.parametrize(
    "mode,coefficient",
    [
        ("mtp_only", 1.0),
        ("mtp_only", 0.3),
        ("joint", 0.0),
        ("joint", 0.1),
        ("joint", 0.3),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("chunks", [0, 1, 4])
def test_distinct_denominators_mixed_gradients_and_empty_target(
    mode, coefficient, empty, chunks
):
    torch.manual_seed(42)
    main = torch.randn(8, 5, requires_grad=mode == "joint")
    auxiliary = torch.randn(8, 5, requires_grad=True)
    head = torch.nn.Linear(5, 16, bias=False).requires_grad_(mode == "joint")
    reference_head = torch.nn.Linear(5, 16, bias=False).requires_grad_(mode == "joint")
    reference_head.load_state_dict(head.state_dict())
    reference = tuple(
        hidden.detach().clone().requires_grad_(hidden.requires_grad)
        for hidden in (main, auxiliary)
    )
    labels = (
        torch.tensor([-100, 2, 3, -100, 5, 6, 7, -100]),
        torch.full((8,), -100)
        if empty
        else torch.tensor([2, 3, -100, 5, 6, -100, -100, -100]),
    )
    counts = tuple((label != -100).sum() for label in labels)
    expected, _, _ = objective_reference(
        *(reference_head(hidden) for hidden in reference),
        *labels,
        mode=mode,
        coefficient=coefficient,
    )
    config = LoopMTPLoss.Config(
        training_mode=mode, loss_scaling_factor=coefficient, global_vocab_size=16
    )
    if chunks:
        loss = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(num_chunks=chunks, loss_fn=config)
        )
        loss.set_lm_head(head)
        predictions = (main, auxiliary)
    else:
        loss = config.build()
        predictions = tuple(head(hidden) for hidden in (main, auxiliary))
    value, metrics = loss(
        predictions, labels, counts[0], mtp_global_valid_tokens=counts[1]
    )
    torch.testing.assert_close(value, expected, atol=1e-5, rtol=1e-4)
    value.backward()
    expected.backward()
    for actual, expected_hidden in zip((main, auxiliary), reference):
        torch.testing.assert_close(
            actual.grad, expected_hidden.grad, atol=1e-5, rtol=1e-4
        )
    torch.testing.assert_close(
        head.weight.grad, reference_head.weight.grad, atol=1e-5, rtol=1e-4
    )
    assert (
        metrics["main_tokens/sum"] == counts[0]
        and metrics["mtp_tokens/sum"] == counts[1]
    )
    assert torch.isfinite(value)
    if empty:
        assert auxiliary.grad is not None and torch.count_nonzero(auxiliary.grad) == 0


def test_fp64_pure_normalization():
    main = torch.tensor(11.3, dtype=torch.float64, requires_grad=True)
    auxiliary = torch.tensor(7.4, dtype=torch.float64, requires_grad=True)
    value, _, _ = normalize_mtp_losses(
        main,
        auxiliary,
        main_count=torch.tensor(3),
        mtp_count=torch.tensor(7),
        training_mode="joint",
        coefficient=0.3,
    )
    torch.testing.assert_close(
        value, main / 3 + 0.3 * auxiliary / 7, atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize(
    "mode,coefficient",
    [
        ("mtp_only", 1.0),
        ("mtp_only", 0.3),
        ("joint", 0.0),
        ("joint", 0.1),
        ("joint", 0.3),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_fp64_objective_and_closed_form_gradients(mode, coefficient, empty):
    torch.manual_seed(42)
    logits = [
        torch.randn(3, 5, dtype=torch.float64, requires_grad=True) for _ in range(2)
    ]
    labels = [
        torch.tensor([-100, 1, 2]),
        torch.tensor([-100, -100, -100] if empty else [3, -100, -100]),
    ]
    counts = [(target != -100).sum() for target in labels]
    sums = [
        torch.nn.functional.cross_entropy(value, target, reduction="sum")
        for value, target in zip(logits, labels)
    ]
    actual, _, _ = normalize_mtp_losses(
        *sums,
        main_count=counts[0],
        mtp_count=counts[1],
        training_mode=mode,
        coefficient=coefficient,
    )
    expected = torch.zeros((), dtype=torch.float64)
    expected_gradients = []
    for axis, (value, target) in enumerate(zip(logits, labels)):
        valid = target != -100
        scale = coefficient if axis == 1 else float(mode == "joint")
        probabilities = value.detach().softmax(-1)
        gradient = probabilities.clone()
        gradient[valid, target[valid]] -= 1
        gradient[~valid] = 0
        gradient *= scale / max(1, int(counts[axis]))
        expected_gradients.append(
            None if axis == 0 and mode == "mtp_only" else gradient
        )
        expected += (
            -scale
            * probabilities[valid, target[valid]].log().sum()
            / counts[axis].clamp_min(1)
        )
    gradients = torch.autograd.grad(actual, logits, allow_unused=True)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("accumulation", [1, 2, 4])
@pytest.mark.parametrize("chunks", [0, 1, 4])
@pytest.mark.parametrize(
    "mode,coefficient",
    [
        ("mtp_only", 1.0),
        ("mtp_only", 0.3),
        ("joint", 0.0),
        ("joint", 0.1),
        ("joint", 0.3),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_logical_batch_uneven_counts_across_accumulation_and_chunks(
    accumulation, chunks, mode, coefficient, empty
):
    torch.manual_seed(43)
    main = torch.randn(32, 5, requires_grad=mode == "joint")
    auxiliary = torch.randn(32, 5, requires_grad=True)
    head = torch.nn.Linear(5, 16, bias=False).requires_grad_(mode == "joint")
    reference_head = torch.nn.Linear(5, 16, bias=False).requires_grad_(mode == "joint")
    reference_head.load_state_dict(head.state_dict())
    references = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        for value in (main, auxiliary)
    )
    main_labels = torch.tensor(
        [
            -100,
            2,
            3,
            -100,
            5,
            6,
            7,
            -100,
            -100,
            -100,
            -100,
            1,
            2,
            -100,
            -100,
            -100,
            1,
            2,
            3,
            4,
            5,
            6,
            -100,
            -100,
            3,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
        ]
    )
    mtp_labels = torch.tensor(
        [
            2,
            3,
            -100,
            5,
            6,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            2,
            -100,
            -100,
            -100,
            -100,
            2,
            3,
            4,
            5,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
        ]
    )
    if empty:
        mtp_labels.fill_(-100)
    counts = ((main_labels != -100).sum(), (mtp_labels != -100).sum())
    expected, main_mean, mtp_mean = objective_reference(
        *(reference_head(value) for value in references),
        main_labels,
        mtp_labels,
        mode=mode,
        coefficient=coefficient,
    )
    expected.backward()
    config = LoopMTPLoss.Config(
        training_mode=mode, loss_scaling_factor=coefficient, global_vocab_size=16
    )
    loss = config.build()
    if chunks:
        loss = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(num_chunks=chunks, loss_fn=config)
        )
        loss.set_lm_head(head)
    total = torch.tensor(0.0)
    metrics = {}
    for index in range(accumulation):
        span = slice(index * (32 // accumulation), (index + 1) * (32 // accumulation))
        values = (main[span], auxiliary[span])
        predictions = values if chunks else tuple(head(value) for value in values)
        objective, part = loss(
            predictions,
            (main_labels[span], mtp_labels[span]),
            counts[0],
            mtp_global_valid_tokens=counts[1],
        )
        objective.backward()
        total += objective.detach()
        for key, value in part.items():
            metrics[key] = metrics.get(key, 0) + value
    torch.testing.assert_close(total, expected, atol=1e-5, rtol=1e-4)
    for value, reference in zip((main, auxiliary), references):
        torch.testing.assert_close(value.grad, reference.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        head.weight.grad, reference_head.weight.grad, atol=1e-5, rtol=1e-4
    )
    assert (
        metrics["main_tokens/sum"] == counts[0]
        and metrics["mtp_tokens/sum"] == counts[1]
    )
    observed_main = metrics["main_loss/sum"] / counts[0]
    observed_mtp = metrics["mtp_loss/sum"] / counts[1].clamp_min(1)
    torch.testing.assert_close(observed_main, main_mean, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(observed_mtp, mtp_mean, atol=1e-5, rtol=1e-4)
    logged = coefficient * observed_mtp + (observed_main if mode == "joint" else 0)
    torch.testing.assert_close(total, logged, atol=1e-6, rtol=1e-6)
