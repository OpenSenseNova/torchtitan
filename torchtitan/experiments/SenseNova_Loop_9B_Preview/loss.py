# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Loop MTP objectives with independent optimizer-step token denominators."""

from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.components.loss import CrossEntropyLoss, IGNORE_INDEX
from torchtitan.distributed.spmd_types import current_spmd_mesh

from torchtitan.experiments.SenseNova_Loop_9B_Preview.model_config import (
    resolve_mtp_config,
)


def normalize_mtp_losses(
    main_sum, mtp_sum, *, main_count, mtp_count, training_mode, coefficient
):
    main = main_sum / main_count.clamp_min(1)
    mtp = mtp_sum / mtp_count.clamp_min(1)
    objective = coefficient * mtp
    if training_mode == "joint":
        objective = objective + main
    return objective, main, mtp


class LoopMTPLoss(CrossEntropyLoss):
    """Monitor the main objective and optimize MTP alone unless joint is explicit."""

    @dataclass(kw_only=True, slots=True)
    class Config(CrossEntropyLoss.Config):
        training_mode: str = "mtp_only"
        loss_scaling_factor: float | None = None

    def __init__(self, config, *, compile_config=None):
        super().__init__(config, compile_config=compile_config)
        settings = resolve_mtp_config(
            num_layers=1,
            training_mode=config.training_mode,
            loss_scaling_factor=config.loss_scaling_factor,
        )
        self.training_mode = settings["training_mode"]
        self.coefficient = settings["loss_scaling_factor"]

    def __call__(
        self,
        pred,
        labels,
        global_valid_tokens=None,
        *,
        mtp_global_valid_tokens=None,
        **kwargs,
    ):
        if not (
            isinstance(pred, tuple)
            and isinstance(labels, tuple)
            and len(pred) == len(labels) == 2
        ):
            raise ValueError(
                "Loop MTP loss requires aligned (main, MTP) prediction and label pairs"
            )
        if global_valid_tokens is None or mtp_global_valid_tokens is None:
            raise ValueError(
                "Loop MTP loss requires both optimizer-step token denominators"
            )
        counts = (global_valid_tokens, mtp_global_valid_tokens)
        if any(
            not isinstance(count, torch.Tensor) or count.ndim != 0 for count in counts
        ):
            raise ValueError("Loop MTP denominators must be scalar device tensors")
        if current_spmd_mesh() is not None:
            for count in counts:
                spmd.assert_type(count, {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I})
        if self.training_mode == "mtp_only":
            with torch.no_grad():
                main_sum, _ = super().__call__(pred[0].detach(), labels[0])
        else:
            main_sum, _ = super().__call__(pred[0], labels[0])
        mtp_sum, _ = super().__call__(pred[1], labels[1])
        objective, _, _ = normalize_mtp_losses(
            main_sum,
            mtp_sum,
            main_count=global_valid_tokens,
            mtp_count=mtp_global_valid_tokens,
            training_mode=self.training_mode,
            coefficient=self.coefficient,
        )
        # A sum CE over ignored labels is a connected zero, so empty ranks
        # still participate in backward collectives. Step skipping is an engine policy.
        metrics = {
            "main_loss/sum": main_sum.detach(),
            "mtp_loss/sum": mtp_sum.detach(),
            "main_tokens/sum": (labels[0] != IGNORE_INDEX).sum(dtype=torch.int64),
            "mtp_tokens/sum": (labels[1] != IGNORE_INDEX).sum(dtype=torch.int64),
        }
        return objective, metrics
