# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Exact future supervision using hand fixtures and independent document IDs."""

import copy
import numpy as np
import pytest
import torch

from torchtitan.experiments.loop_qwen35.mtp_data import (
    MTPMicrobatch,
    align_mtp_batch,
    vision_bank_indices,
)
from mtp_reference import align_reference, load_cases


@pytest.mark.parametrize(
    "case", load_cases()["alignment"], ids=lambda case: case["name"]
)
def test_literal_and_independent_alignment(case):
    batch = {
        key: torch.tensor(value)
        for key, value in case.items()
        if key not in ("name", "documents", "expected")
    }
    before = copy.deepcopy(batch)
    actual = align_mtp_batch(batch)
    independent = align_reference(case)
    assert independent == case["expected"]
    assert actual.keys() == independent.keys()
    for name, expected in independent.items():
        assert actual[name].tolist() == expected
    assert (
        actual["mtp_valid_tokens"].dtype == torch.int64
        and actual["mtp_valid_tokens"].ndim == 0
    )
    assert all(torch.equal(value, before[key]) for key, value in batch.items())


@pytest.mark.parametrize("cp,tp", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_align_before_cp_and_sp_boundaries(cp, tp):
    case = {
        "input": list(range(1, 17)),
        "labels": list(range(2, 18)),
        "positions": list(range(11)) + list(range(5)),
        "documents": [0] * 11 + [1] * 5,
    }
    case["labels"][10] = case["labels"][15] = -100
    actual = align_mtp_batch(
        {key: torch.tensor(value) for key, value in case.items() if key != "documents"}
    )
    reference = align_reference(case)
    for rank in range(cp * tp):
        span = slice(rank * (16 // (cp * tp)), (rank + 1) * (16 // (cp * tp)))
        assert actual["mtp_labels"][span].tolist() == reference["mtp_labels"][span]
        assert actual["mtp_input"][span].tolist() == reference["mtp_input"][span]
    assert actual["mtp_labels"][7] == 10 and actual["mtp_input"][7] == 9
    assert actual["mtp_input"][10] == 0


def test_image_and_video_bank_order_is_independent_of_text_order():
    batch = {
        "input": torch.tensor([7, 43, 8, 42, 42, 9]),
        "labels": torch.tensor([-100, 8, -100, -100, 9, -100]),
        "positions": torch.arange(6),
        "special_tokens": {"image_id": 42, "video_id": 43},
        "pixel_values": torch.ones(2, 4),
        "pixel_values_videos": torch.ones(1, 4),
    }
    assert vision_bank_indices(batch).tolist() == [-1, 2, -1, 0, 1, -1]
    assert align_mtp_batch(batch)["mtp_vision_bank_indices"].tolist() == [
        2,
        -1,
        0,
        1,
        -1,
        -1,
    ]


def boundary_loader(length, strategy):
    from test_balanced_packing import loader as make_loader

    if strategy == "balanced":
        # Strict planning rejects positive over-limit lengths. Zero records an
        # offline exclusion; online encoding must not silently change the plan.
        with pytest.raises(ValueError, match="offline length"):
            make_loader([length + 1, length], budget=length)
    loader = make_loader(
        [length + 1, length],
        estimated=[0, length] if strategy == "balanced" else None,
        budget=length,
        strategy=strategy,
    )
    if strategy == "greedy":
        loader.order = np.array([0, 1])
    return loader


@pytest.mark.parametrize("strategy", ["greedy", "balanced"])
def test_exact_65536_kept_and_overlength_skipped_without_changing_main_labels(strategy):
    loader = boundary_loader(65536, strategy)
    batch = next(loader)
    assert loader.skipped == (1 if strategy == "greedy" else 0)
    assert batch.input.numel() == 65536 and not batch.padding_mask.any()
    assert batch.num_valid_tokens == 65535
    expected_main = torch.cat((batch.input[1:], torch.tensor([-100])))
    assert torch.equal(batch.labels, expected_main)
    state = (loader.cursor, loader.epoch, loader.skipped)
    original = batch.labels.clone()
    auxiliary = align_mtp_batch(batch.as_input_dict())
    assert auxiliary["mtp_valid_tokens"] == 65534
    assert auxiliary["mtp_labels"][-2:].tolist() == [-100, -100]
    assert torch.equal(auxiliary["mtp_labels"][:-2], batch.input[2:])
    wrapped = MTPMicrobatch(
        source=batch, mtp_global_valid_tokens=auxiliary["mtp_valid_tokens"]
    )
    assert wrapped.labels is batch.labels and torch.equal(wrapped.labels, original)
    assert (loader.cursor, loader.epoch, loader.skipped) == state


def test_mrope_cannot_replace_document_identity():
    with pytest.raises(ValueError, match="reset positions"):
        align_mtp_batch(
            {
                "input": torch.arange(4),
                "labels": torch.arange(4),
                "positions": torch.zeros(4, 3),
            }
        )
