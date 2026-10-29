# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-independent future-token alignment for packed MTP sequences."""

import spmd_types as spmd
import torch


def roll_mtp_sequence(
    sequence: torch.Tensor,
    *,
    shift: int,
    fill_value: int,
    positions: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    return_valid_mask: bool = False,
    strict_padding: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Shift ``[T, ...]`` left without wrapping or crossing document boundaries.

    Reset-style ``positions[T]`` identify documents. ``strict_padding``
    fills both padded sources and destinations. The default preserves the
    DeepSeek helper's behavior, including padded source values in calls
    without positions (their validity mask remains false). The optional
    validity mask has the same shape as ``sequence``; token validity is
    broadcast over all trailing dimensions.
    """
    seq_len = sequence.shape[0]
    if shift <= 0 or shift > seq_len:
        raise ValueError(f"MTP roll shift must be in [1, {seq_len}], got {shift}.")
    if positions is not None and (positions.ndim != 1 or positions.shape[0] < seq_len):
        raise ValueError(
            f"MTP positions need at least {seq_len} reset-style token positions."
        )
    if padding_mask is not None and (
        padding_mask.ndim != 1 or padding_mask.shape[0] < seq_len
    ):
        raise ValueError(f"MTP padding mask needs at least {seq_len} token positions.")

    source = sequence[shift:]
    valid_tokens = torch.ones(seq_len - shift, dtype=torch.bool, device=sequence.device)
    if positions is not None:
        valid_tokens = positions[shift:seq_len] == positions[: seq_len - shift] + shift
    if padding_mask is not None:
        valid_tokens = valid_tokens & ~padding_mask[shift:seq_len]
        if strict_padding:
            valid_tokens = valid_tokens & ~padding_mask[: seq_len - shift]
    rolled = torch.full_like(sequence, fill_value)
    if positions is None and not strict_padding:
        # Legacy position-free callers retain source values at masked slots.
        rolled[: seq_len - shift] = source
    valid_mask = torch.zeros_like(sequence, dtype=torch.bool)
    shape = (seq_len - shift,) + (1,) * (sequence.ndim - 1)
    with spmd.no_typecheck():
        valid_mask[: seq_len - shift] = valid_tokens.reshape(shape).expand_as(source)
    rolled[: seq_len - shift] = torch.where(
        valid_mask[: seq_len - shift], source, rolled[: seq_len - shift]
    )
    return (rolled, valid_mask) if return_valid_mask else rolled
