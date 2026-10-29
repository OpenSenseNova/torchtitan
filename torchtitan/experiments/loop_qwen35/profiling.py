# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded, opt-in profiler capture; never launches a training process."""

from pathlib import Path

import torch


def sample_step(step, interval):
    if interval < 0:
        raise ValueError("Diagnostic interval must be nonnegative")
    return interval > 0 and step % interval == 0


class StepProfiler:
    def __init__(
        self,
        *,
        directory,
        rank,
        ranks,
        start_step,
        num_steps,
        device_type,
        with_stack=True,
        profile_memory=True,
    ):
        self.directory = Path(directory)
        self.rank = rank
        self.ranks = ranks
        self.start_step = start_step
        self.num_steps = num_steps
        self.device_type = device_type
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.step_context = None
        self.profiler = None
        self.last_step = None

    @staticmethod
    def _sync_ranks():
        # Profiler setup/export may take minutes on one rank. Keep that work
        # outside every rank's training timer, including the following update.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def contains(self, step):
        return (
            self.start_step > 0
            and self.start_step <= step < self.start_step + self.num_steps
        )

    def begin(self, step):
        if self.rank in self.ranks and step == self.start_step and self.contains(step):
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device_type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
            )
            self.profiler.start()
        if self.profiler is not None:
            self.last_step = step
        if step == self.start_step and self.contains(step):
            self._sync_ranks()
        if self.profiler is not None:
            self.step_context = torch.profiler.record_function(f"train_step_{step}")
            self.step_context.__enter__()

    def end(self, step):
        self._end_step_context()
        if self.profiler is not None:
            self.profiler.step()
            if step == self.start_step + self.num_steps - 1:
                self.close()
        if self.contains(step) and step == self.start_step + self.num_steps - 1:
            self._sync_ranks()

    def _end_step_context(self):
        if self.step_context is not None:
            context, self.step_context = self.step_context, None
            context.__exit__(None, None, None)

    def close(self):
        self._end_step_context()
        if self.profiler is not None:
            profiler, self.profiler = self.profiler, None
            profiler.stop()
            self.directory.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(
                str(
                    self.directory
                    / f"rank-{self.rank}-steps-{self.start_step}-{self.last_step}.json"
                )
            )
