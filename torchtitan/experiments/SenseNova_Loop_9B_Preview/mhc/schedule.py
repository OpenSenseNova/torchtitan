# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Independent, checkpointable backbone and router annealing schedules."""

import math
from dataclasses import dataclass

from torch.optim.lr_scheduler import LambdaLR

from torchtitan.components.optimizer import LRSchedulersContainer


def factor(step, total, warmup, minimum, decay_ratio=1.0):
    if step < warmup:
        return (step + 1) / warmup
    if decay_ratio == 0:
        return 1.0
    decay_steps = max(1, round((total - warmup) * decay_ratio))
    decay_start = total - decay_steps
    progress = min(1.0, max(0.0, (step - decay_start) / decay_steps))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


class MHCSchedulers(LRSchedulersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(LRSchedulersContainer.Config):
        router_warmup_steps: int = 20
        decay_ratio: float = 1.0

        def build(self, *, optimizers, training_steps):
            total = self.total_steps or training_steps
            if (
                not 0 <= self.warmup_steps < total
                or not 0 <= self.router_warmup_steps < total
            ):
                raise ValueError(
                    "Both warmup durations must be nonnegative and below total steps"
                )
            return MHCSchedulers(
                optimizers,
                total,
                self.warmup_steps,
                self.router_warmup_steps,
                self.min_lr_factor,
                self.decay_ratio,
            )

    def __init__(
        self,
        optimizers,
        total,
        backbone_warmup,
        router_warmup,
        minimum,
        decay_ratio=1.0,
    ):
        self.schedulers = []
        for optimizer in optimizers:
            lambdas = []
            for group in optimizer.param_groups:
                names = group["param_names"]
                kinds = {".mhc_" in name for name in names}
                if len(kinds) != 1:
                    raise ValueError(
                        "Backbone and router must use separate optimizer groups"
                    )
                warmup = router_warmup if True in kinds else backbone_warmup
                lambdas.append(
                    lambda step, w=warmup: factor(step, total, w, minimum, decay_ratio)
                )
            self.schedulers.append(LambdaLR(optimizer, lambdas))

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Also restore actual optimizer group LR before the first resumed update.
        for scheduler in self.schedulers:
            for group, lr in zip(
                scheduler.optimizer.param_groups, scheduler._last_lr, strict=True
            ):
                group["lr"] = lr

    def get_metrics(self):
        metrics = {}
        for scheduler in self.schedulers:
            for group, lr in zip(
                scheduler.optimizer.param_groups, scheduler.get_last_lr(), strict=True
            ):
                label = "router" if ".mhc_" in group["param_names"][0] else "backbone"
                metrics["lr/" + label] = float(lr)
        return metrics
