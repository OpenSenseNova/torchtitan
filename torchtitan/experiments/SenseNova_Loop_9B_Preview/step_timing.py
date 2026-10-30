# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Per-step phase observations without synchronizing CUDA at phase boundaries."""

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist


class StepTiming:
    CPU_PHASES = (
        "dataloader",
        "data_local",
        "data_sync",
        "prepare",
        "encode_wait",
        "collate_text",
        "collate_images",
        "collate_mrope",
        "plan_epoch",
        "pack_wait",
        "prefetch_data_local",
        "prefetch_encode_wait",
        "prefetch_collate_text",
        "prefetch_collate_images",
        "prefetch_collate_mrope",
    )
    GPU_PHASES = (
        "microbatch",
        "forward",
        "loss",
        "backward",
        "optimizer",
        "multimodal",
        "vision",
        "decoder",
    )
    WORK = ("microbatches", "nonpadding_tokens", "documents", "attention_cost")

    def __init__(self, device, *, enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.active = False
        self.steps = 0
        self.host = {name: 0.0 for name in self.CPU_PHASES + self.GPU_PHASES}
        self.work = {name: 0.0 for name in self.WORK}
        self.events = []
        self.last_rank_values = []

    def begin_step(self):
        if self.enabled:
            if self.active:
                raise RuntimeError("Timing step already active")
            self.active = True
            self.steps += 1

    def end_step(self):
        self.active = False

    @contextmanager
    def phase(self, name):
        if not self.active:
            yield
            return
        if name not in self.host:
            raise ValueError(f"Unknown training phase: {name}")
        gpu = self.device.type == "cuda" and name in self.GPU_PHASES
        if gpu:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.host[name] += time.perf_counter() - started
            if gpu:
                end.record()
                self.events.append((name, start, end))

    def record_batch(self, lengths, budget):
        if self.active:
            tokens = sum(lengths)
            self.work["microbatches"] += 1
            self.work["nonpadding_tokens"] += tokens
            self.work["documents"] += len(lengths)
            self.work["attention_cost"] += (
                sum(n * n for n in lengths) + (budget - tokens) ** 2
            )

    def collect_interval(self):
        """Aggregate rank-local interval sums, normalized by observed steps.

        Events measure current-stream elapsed intervals, including dependent
        communication waits, not a sum of kernel compute times. Nested phases
        (microbatch vs forward/loss/backward, dataloader vs data_local/data_sync)
        must not be added. Maxima can come from different ranks.
        """
        if not self.enabled or not self.steps:
            return {}
        started = time.perf_counter()
        gpu = {name: 0.0 for name in self.GPU_PHASES}
        if self.events:
            # Read at the existing metrics boundary, never between phases.
            # All events are recorded on the trainer's current compute stream.
            last = self.events[-1][2]
            if not last.query():
                last.synchronize()
            for name, start, end in self.events:
                gpu[name] += start.elapsed_time(end) / 1000.0
        values = {
            f"{name}_host_s": value / self.steps for name, value in self.host.items()
        }
        if self.device.type == "cuda":
            values.update(
                {f"{name}_gpu_s": value / self.steps for name, value in gpu.items()}
            )
        values.update(
            {
                f"{name}_per_step": value / self.steps
                for name, value in self.work.items()
            }
        )
        keys = tuple(values)
        local = torch.tensor(
            [self.steps, *values.values()], dtype=torch.float64, device=self.device
        )
        if dist.is_initialized():
            gathered = torch.empty(
                dist.get_world_size() * local.numel(),
                dtype=local.dtype,
                device=self.device,
            )
            dist.all_gather_into_tensor(gathered, local)
            rows = gathered.reshape(-1, local.numel()).cpu()
        else:
            rows = local.unsqueeze(0).cpu()
        if not (rows[:, 0] == self.steps).all():
            raise RuntimeError("Training timing interval differs across ranks")
        self.last_rank_values = [dict(zip(keys, row[1:].tolist())) for row in rows]
        result = {"step_timing/steps": self.steps}
        for index, key in enumerate(keys, 1):
            column = rows[:, index]
            result.update(
                {
                    f"step_timing/{key}/min": column.min().item(),
                    f"step_timing/{key}/mean": column.mean().item(),
                    f"step_timing/{key}/max": column.max().item(),
                    f"step_timing/{key}/max_rank": int(column.argmax().item()),
                }
            )
        # Local collector cost includes any completion wait and the collective;
        # it is an instrumentation overhead, not a model phase.
        result["step_timing/collector_host_s"] = time.perf_counter() - started
        self.steps = 0
        self.host = dict.fromkeys(self.host, 0.0)
        self.work = dict.fromkeys(self.work, 0.0)
        self.events.clear()
        return result


def timing_log_line(step, metrics):
    """Compact rank-0 output; complete min/mean/max values remain in metrics."""
    device = "gpu" if "step_timing/forward_gpu_s/mean" in metrics else "host"
    parts = []
    for name in (
        "dataloader",
        "data_local",
        "encode_wait",
        "collate_text",
        "collate_images",
        "collate_mrope",
        "pack_wait",
        "data_sync",
        "plan_epoch",
        "prepare",
        "forward",
        "multimodal",
        "vision",
        "decoder",
        "loss",
        "backward",
        "optimizer",
        "microbatch",
    ):
        kind = "host" if name in StepTiming.CPU_PHASES else device
        prefix = f"step_timing/{name}_{kind}_s/"
        parts.append(
            f"{name}_{kind}={metrics[prefix + 'mean']:.3f}/"
            f"{metrics[prefix + 'max']:.3f}s@r{metrics[prefix + 'max_rank']}"
        )
    for name in ("nonpadding_tokens", "documents", "attention_cost"):
        prefix = f"step_timing/{name}_per_step/"
        parts.append(
            f"{name}={metrics[prefix + 'mean']:.0f}/{metrics[prefix + 'max']:.0f}"
        )
    return (
        f"step_timing step={step} steps={metrics['step_timing/steps']} "
        "per_step_mean/max: " + " ".join(parts)
    )
