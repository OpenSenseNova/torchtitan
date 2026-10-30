# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Align MTP supervision on complete microbatches before CP or TP slicing."""

from dataclasses import dataclass

import torch

from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.common.mtp import roll_mtp_sequence


def vision_bank_indices(batch):
    """Match the shared image-then-video bank order before token sharding."""
    from torchtitan.models.common.multimodal import build_vision_bank_indices

    if "vision_bank_indices" in batch:
        return batch["vision_bank_indices"]
    tokens = batch["input"]
    indices = torch.full_like(tokens, -1)
    offset = torch.zeros((), device=tokens.device, dtype=torch.int64)
    for pixel_key, token_key in (
        ("pixel_values", "image_id"),
        ("pixel_values_videos", "video_id"),
    ):
        pixels = batch.get(pixel_key)
        if pixels is None or pixels.shape[0] == 0:
            continue
        if batch.get("special_tokens") is None:
            raise ValueError("special_tokens is required for MTP vision inputs")
        local = build_vision_bank_indices(
            tokens, placeholder_id=batch["special_tokens"][token_key]
        )
        indices = torch.where(local >= 0, local + offset, indices)
        offset = offset + (local >= 0).sum()
    return indices


def align_mtp_batch(batch):
    """Return future inputs and already-shifted future labels without mutation.

    Loader labels already predict one token ahead. Shifting them once here
    predicts two ahead; the current label mask must not intersect this mask.
    Document identity always comes from reset positions, never MRoPE axes.
    """
    tokens, labels, positions = (batch[key] for key in ("input", "labels", "positions"))
    if (
        tokens.ndim != 1
        or labels.shape != tokens.shape
        or positions.shape != tokens.shape
    ):
        raise ValueError("MTP requires aligned flat input, labels and reset positions")
    padding = batch.get("padding_mask")
    if padding is not None and padding.shape != tokens.shape:
        raise ValueError("MTP padding mask must match the full token sequence")
    options = dict(
        shift=1, positions=positions, padding_mask=padding, strict_padding=True
    )
    future_tokens, valid = roll_mtp_sequence(
        tokens, fill_value=0, return_valid_mask=True, **options
    )
    future_labels = roll_mtp_sequence(labels, fill_value=IGNORE_INDEX, **options)
    rope_positions = batch.get("mrope_positions", positions)
    result = {
        "mtp_input": future_tokens,
        "mtp_labels": future_labels,
        "mtp_input_valid_mask": valid,
        "mtp_positions": roll_mtp_sequence(rope_positions, fill_value=0, **options),
        "mtp_valid_tokens": (future_labels != IGNORE_INDEX).sum(dtype=torch.int64),
    }
    if any(
        key in batch
        for key in ("vision_bank_indices", "pixel_values", "pixel_values_videos")
    ):
        result["mtp_vision_bank_indices"] = roll_mtp_sequence(
            vision_bank_indices(batch), fill_value=-1, **options
        )
    return result


@dataclass(kw_only=True, slots=True)
class MTPMicrobatch(TrainingMicrobatch):
    """Attach an optimizer-step denominator without advancing the source loader."""

    source: TrainingMicrobatch
    mtp_global_valid_tokens: torch.Tensor

    @property
    def labels(self):
        return self.source.labels

    @property
    def num_valid_tokens(self):
        return self.source.num_valid_tokens

    def as_input_dict(self):
        return self.source.as_input_dict()

    def loss_kwargs(self):
        return {
            **self.source.loss_kwargs(),
            "mtp_global_valid_tokens": self.mtp_global_valid_tokens,
        }
