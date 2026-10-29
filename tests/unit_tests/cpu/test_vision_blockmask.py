# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packed vision isolation and the opt-in BlockMask compiler."""

import importlib.util
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.models.common import vision_encoder


@pytest.mark.parametrize("mode", [None, "0", "1"])
def test_compile_is_opt_in_at_import(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("VISION_BLOCKMASK_COMPILE", raising=False)
    else:
        monkeypatch.setenv("VISION_BLOCKMASK_COMPILE", mode)
    # Load an isolated module so the selection cannot affect other tests.
    spec = importlib.util.spec_from_file_location(
        "vision_encoder_test", vision_encoder.__file__
    )
    module = importlib.util.module_from_spec(spec)
    with patch("torch.compile") as compile_mask:
        spec.loader.exec_module(module)
    if mode == "1":
        compile_mask.assert_called_once_with(create_block_mask)
        assert module._vision_create_block_mask is compile_mask.return_value
    else:
        compile_mask.assert_not_called()
        assert module._vision_create_block_mask is create_block_mask


@pytest.mark.parametrize("lengths", [[4], [128, 256], [3, 129, 64], [2992] * 3])
def test_packed_images_have_bidirectional_isolated_attention(monkeypatch, lengths):
    monkeypatch.setattr(vision_encoder, "_vision_create_block_mask", create_block_mask)
    total = sum(lengths)
    mask = vision_encoder.create_block_diagonal_mask(
        torch.tensor(lengths), total, torch.device("cpu")
    )
    # Check every valid query/key pair in bounded tiles, including partial blocks.
    boundaries = torch.tensor(lengths).cumsum(0)
    keys = torch.arange(total)
    key_segments = torch.bucketize(keys, boundaries, right=True)
    for start in range(0, total, 128):
        queries = torch.arange(start, min(start + 128, total))
        expected = (
            torch.bucketize(queries, boundaries, right=True)[:, None]
            == key_segments[None, :]
        )
        actual = mask.mask_mod(None, None, queries[:, None], keys[None, :])
        assert torch.equal(actual, expected)
