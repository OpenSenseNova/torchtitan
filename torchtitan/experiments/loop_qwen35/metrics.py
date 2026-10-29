# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Useful decoder FLOPs for packed Loop training, excluding padding and recompute.

Use the native dense-attention MFU convention (no causal sparsity discount).
Count decoder projections, GDN state updates and mHC routing matmuls, including
every loop execution. Vision, elementwise ops and activation recompute are not
part of this decoder MFU estimate.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from torchtitan.models.utils import (
    delta_rule_flops_per_token,
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.observability.metrics import (
    compute_training_performance_metrics,
    MetricsProcessor,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopFlops:
    nparams: int
    linear: int
    quadratic: int
    loop_linear: int
    loop_quadratic: int

    @classmethod
    def from_model(cls, model, config):
        nparams, active = get_nparams_and_active_nparams(
            model, modules_excluded_from_active_params=(model.vision_encoder,)
        )
        linear, quadratic, loop_linear, loop_quadratic = 6 * active, 0, 0, 0
        for index, layer_config in enumerate(config.layers):
            layer = model.layers[str(index)]
            # These dense Qwen3.5 blocks include mHC projection parameters.
            layer_linear = 6 * sum(p.numel() for p in layer.parameters())
            extra, attention = 0, 0
            if layer_config.attention is not None:
                cfg = layer_config.attention
                attention = quadratic_attention_flops_per_token(
                    num_heads=cfg.n_heads,
                    qk_head_dim=cfg.head_dim,
                    v_head_dim=cfg.head_dim,
                    seq_len=1,
                )
            if layer_config.delta_net is not None:
                cfg = layer_config.delta_net
                extra += delta_rule_flops_per_token(
                    num_heads=cfg.in_proj_v.out_features // cfg.value_head_dim,
                    key_head_dim=cfg.key_head_dim,
                    v_head_dim=cfg.value_head_dim,
                )
            if hasattr(layer, "mhc_streams"):
                streams = layer.mhc_streams
                # Two routers per block: pre contraction and residual mixing.
                extra += 2 * 6 * config.dim * (streams + streams * streams)
            linear += extra
            quadratic += attention
            if config.loop.start <= index < config.loop.end:
                loop_linear += layer_linear + extra
                loop_quadratic += attention
        return cls(nparams, linear, quadratic, loop_linear, loop_quadratic)

    def batch(self, lengths, repeat):
        return (self.linear + (repeat - 1) * self.loop_linear) * sum(lengths) + (
            self.quadratic + (repeat - 1) * self.loop_quadratic
        ) * sum(length * length for length in lengths)


class LoopMetricsProcessor(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._counts = [0.0, 0.0, 0.0, 0.0]
        self._started = False
        self._checkpoint_seconds = 0.0
        self._interval_steps = 1
        self.device = self.device_memory_monitor.device
        logger.info(
            "MFU scope: useful decoder FLOPs, actual packed document lengths and loop count; "
            "includes mHC matmuls, excludes padding, vision, elementwise ops and recompute"
        )
        logger.info(
            "Throughput unit: tokens/s/GPU. TGS includes padding; valid_TGS excludes padding. "
            "Both exclude checkpoint save time and include data loading, forward/backward and optimizer work."
        )

    def begin_step(self):
        if not self._started:
            # Exclude model/data initialization and checkpoint loading on fresh/resumed runs.
            self.time_last_log = time.perf_counter()
            self._started = True

    @contextmanager
    def exclude_time(self):
        """Exclude profiler setup/export without resetting a multi-step interval."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.time_last_log += duration

    def record_batch(self, lengths, repeat, counter):
        self._counts[0] += sum(lengths)
        self._counts[1] += counter.batch(lengths, repeat)
        self._counts[2] += len(lengths)
        self._counts[3] += not lengths

    def get_interval_metrics(self):
        timer = getattr(self, "step_timing", None)
        timing_metrics = timer.collect_interval() if timer is not None else {}
        # Include the optional timing collector in the observed wall time.
        elapsed = time.perf_counter() - self.time_last_log
        # Each CP/TP peer sees the same unsharded input. Averaging over all
        # ranks gives a DP-rank mean; divide by TP*CP exactly once below.
        counts = torch.tensor(
            [*self._counts, self.ntokens_since_last_log],
            dtype=torch.float64,
            device=self.device,
        )
        duration = torch.tensor(
            [
                elapsed,
                max(elapsed - self._checkpoint_seconds, 1e-9),
                self._checkpoint_seconds,
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_initialized():
            dist.all_reduce(counts)
            counts /= dist.get_world_size()
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        real_tokens, flops, documents, empty, slots = counts.tolist()
        elapsed, train_elapsed, checkpoint_elapsed = duration.tolist()
        non_dp = self.parallel_dims.non_data_parallel_size
        performance = compute_training_performance_metrics(
            num_tokens=slots,
            elapsed_time=elapsed,
            non_data_parallel_size=non_dp,
            num_flops_per_token=flops / slots,
            gpu_peak_flops=self.gpu_peak_flops,
            has_quantization=self.has_quantization,
        )
        metrics = {
            "throughput(TGS)": slots / (train_elapsed * non_dp),
            "throughput(valid_TGS)": real_tokens / (train_elapsed * non_dp),
            "time_metrics/train(s)": train_elapsed / self._interval_steps,
            "time_metrics/checkpoint(s)": checkpoint_elapsed / self._interval_steps,
            "time_metrics/train_interval(s)": train_elapsed,
            "time_metrics/checkpoint_interval(s)": checkpoint_elapsed,
            "tflops": performance["tflops"],
            "packing/padding(%)": 100 * (slots - real_tokens) / slots,
            "packing/documents_per_dp_rank": documents,
            "packing/empty_microbatches_per_dp_rank": empty,
            **timing_metrics,
        }
        if "mfu_percent" in performance:
            metrics["mfu(%)"] = performance["mfu_percent"]
        self._counts = [0.0, 0.0, 0.0, 0.0]
        self._checkpoint_seconds = 0.0
        return elapsed, metrics

    def format_throughput(self, metrics):
        return (
            f"TGS: {metrics['throughput(TGS)']:,.1f}  "
            f"valid_TGS: {metrics['throughput(valid_TGS)']:,.1f} (tokens/s/GPU, excluding checkpoint)  "
            f"padding: {metrics['packing/padding(%)']:.2f}%  "
            f"train_time: {metrics['time_metrics/train(s)']:.3f}s  "
            f"checkpoint_time: {metrics['time_metrics/checkpoint(s)']:.3f}s"
        )

    def log(self, step, *args, **kwargs):
        self._interval_steps = (
            step - self.step_last_log if self.step_last_log is not None else 1
        )
        super().log(step, *args, **kwargs)
        metrics = self.last_metrics
        if "step_timing/steps" in metrics and (
            not dist.is_initialized() or dist.get_rank() == 0
        ):
            from torchtitan.experiments.loop_qwen35.step_timing import timing_log_line

            logger.info(timing_log_line(step, metrics))

    def record_checkpoint(self, seconds):
        self._checkpoint_seconds += seconds
