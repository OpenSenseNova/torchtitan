# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Gradient clipping for explicit local TP plus FSDP parameter shards."""

import json
import logging

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.sharding import resolve_placements
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine

logger = logging.getLogger(__name__)


@torch.no_grad()
def synchronize_replicated_tp_gradients(model_parts, mesh):
    """Sum partial gradients of TP R parameters outside the DP/CP FSDP mesh.

    TP I parameters already have identical gradients and must not be summed.
    Pack the small replicated parameters to avoid one collective per norm.
    """
    buckets, seen = {}, set()
    for model in model_parts:
        for module in model.modules():
            config = getattr(module, "_sharding_config", None)
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad or param.grad is None or id(param) in seen:
                    continue
                seen.add(id(param))
                if config is None or name not in config.state_shardings:
                    raise ValueError(
                        f"Missing TP placement for {type(module).__name__}.{name}"
                    )
                if config.state_shardings[name].local_type.get("tp") is not spmd.R:
                    continue
                grad = (
                    param.grad.to_local()
                    if isinstance(param.grad, DTensor)
                    else param.grad
                )
                buckets.setdefault((grad.device, grad.dtype), []).append(grad)
    for gradients in buckets.values():
        flat = torch.cat([grad.reshape(-1) for grad in gradients])
        dist.all_reduce(flat, group=mesh.get_group())
        offset = 0
        for grad in gradients:
            grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
            offset += grad.numel()


class LoopTrainingEngine(TrainingEngine):
    @property
    def mtp_settings(self):
        parts = getattr(self, "model_parts", ())
        settings = getattr(parts[0], "mtp_settings", None) if parts else None
        return settings if settings and settings["enabled"] else None

    def _audit_mtp_parameters(self, *, optimizer=False):
        if not self.mtp_settings:
            return
        frozen_mode = False
        for model in self.model_parts:
            model.audit_mtp_trainability()
            frozen_mode |= model.mtp_only()
        if not (optimizer and frozen_mode):
            return
        expected = {
            id(p)
            for model in self.model_parts
            for p in model.parameters()
            if p.requires_grad
        }
        actual = [
            p
            for opt in self.optimizers
            for group in opt.param_groups
            for p in group["params"]
        ]
        if {id(p) for p in actual} != expected or len(actual) != len(expected):
            raise ValueError(
                "mtp_only optimizer must contain every trainable MTP parameter exactly once"
            )
        for opt in self.optimizers:
            if any(not p.requires_grad for p in opt.state):
                raise ValueError(
                    "mtp_only optimizer contains state for a frozen parameter"
                )

    def _initialize_optimizer(self, model_spec):
        if not self.mtp_settings:
            return super()._initialize_optimizer(model_spec)
        self.optimizer_updates = 0
        self.skipped_empty_mtp_steps = 0
        self._audit_mtp_parameters()
        from torchtitan.experiments.loop_qwen35.checkpoint import (
            preflight_native_resume,
        )

        preflight_native_resume(self)
        for model in self.model_parts:
            settings = getattr(model, "mtp_settings", None)
            if settings and settings["enabled"]:
                from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import (
                    parameter_manifest,
                )

                manifest = parameter_manifest(model)
                checkpointer = self.config.checkpointer
                if (
                    checkpointer is not None
                    and checkpointer.resolved_config is not None
                ):
                    checkpointer.resolved_config["mtp"].update(manifest)
                logger.info(
                    "Loop MTP mode=%s coefficient=%s hidden_stage=%s parameters=%s",
                    settings["training_mode"],
                    settings["loss_scaling_factor"],
                    settings["hidden_stage"],
                    json.dumps(manifest, sort_keys=True),
                )
        super()._initialize_optimizer(model_spec)
        self._audit_mtp_parameters(optimizer=True)

    def load_checkpoint(self):
        loaded = super().load_checkpoint()
        self._audit_mtp_parameters(optimizer=True)
        return loaded

    def prepare_step(self, global_valid_tokens, *, num_accumulation_steps=1):
        for model in self.model_parts:
            model.set_training_step(self.num_completed_steps)
        denominator = super().prepare_step(
            global_valid_tokens, num_accumulation_steps=num_accumulation_steps
        )
        if self.mtp_settings:
            if not hasattr(self, "mtp_global_valid_tokens"):
                raise ValueError(
                    "MTP step counts must be prepared from all prefetched microbatches"
                )
            torch._assert_async(
                denominator > 0,
                "A training step must contain main supervision globally",
            )
            self.main_global_valid_tokens = denominator
            self.mtp_step_metrics = {}
        return denominator

    def forward_backward_microbatch(self, **kwargs):
        loss = super().forward_backward_microbatch(**kwargs)
        # Release retained MACA 2.8 router weights before updating sharded
        # masters, so the next forward gathers the new optimizer values.
        for model in self.model_parts:
            for module in model.modules():
                if getattr(module, "_c500_reshard_after_microbatch", False):
                    module.reshard()
        if self.mtp_settings:
            # The outer call completes any replay before counting this batch.
            for name, value in self.loss_metrics.items():
                if name in self.mtp_step_metrics:
                    self.mtp_step_metrics[name].add_(value.detach())
                else:
                    dtype = torch.float64 if value.is_floating_point() else torch.int64
                    self.mtp_step_metrics[name] = value.detach().to(dtype=dtype).clone()
        return loss

    @torch.no_grad()
    def mtp_step_statistics(self):
        from torchtitan.experiments.loop_qwen35.loss import normalize_mtp_losses

        metrics = self.mtp_step_metrics
        local_sums = torch.stack((metrics["main_loss/sum"], metrics["mtp_loss/sum"]))
        local_counts = torch.stack(
            (metrics["main_tokens/sum"], metrics["mtp_tokens/sum"])
        )
        settings = self.mtp_settings
        local_objective, _, _ = normalize_mtp_losses(
            *local_sums,
            main_count=local_counts[0],
            mtp_count=local_counts[1],
            training_mode=settings["training_mode"],
            coefficient=settings["loss_scaling_factor"],
        )
        sums, counts = local_sums.clone(), local_counts.clone()
        ntokens = torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device)
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")
        if loss_mesh is not None:
            group = loss_mesh.get_group()
            dist.all_reduce(sums, group=group)
            dist.all_reduce(counts, group=group)
            dist.all_reduce(ntokens, group=group)
            dist.all_reduce(local_objective, op=dist.ReduceOp.MAX, group=group)
        expected = torch.stack(
            (self.main_global_valid_tokens, self.mtp_global_valid_tokens)
        )
        if not torch.equal(counts, expected):
            raise ValueError(
                f"MTP step token counts disagree: loss={counts.tolist()} prefetched={expected.tolist()}"
            )
        objective, main, auxiliary = normalize_mtp_losses(
            *sums,
            main_count=counts[0],
            mtp_count=counts[1],
            training_mode=settings["training_mode"],
            coefficient=settings["loss_scaling_factor"],
        )
        return {
            "loss/main": float(main),
            "loss/mtp": float(auxiliary),
            "loss/train": float(objective),
            "loss/total": float(objective),
            "loss/main_sum": float(sums[0]),
            "loss/mtp_sum": float(sums[1]),
            "loss/max_local_train": float(local_objective),
            "loss/main_role": (
                "monitor" if settings["training_mode"] == "mtp_only" else "objective"
            ),
            "mtp/training_mode": settings["training_mode"],
            "mtp/loss_scaling_factor": settings["loss_scaling_factor"],
            "tokens/main_valid": int(counts[0]),
            "tokens/mtp_valid": int(counts[1]),
            "n_tokens_seen": int(ntokens),
            "steps/consumed": self.num_completed_steps,
            "steps/optimizer_updates": self.optimizer_updates,
            "steps/skipped_empty_mtp": self.skipped_empty_mtp_steps,
        }

    def optimizer_step(self):
        if self.parallel_dims.tp > 1:
            synchronize_replicated_tp_gradients(
                self.model_parts, self.parallel_dims.get_mesh("tp")
            )
        settings = self.mtp_settings
        if settings is None:
            return super().optimizer_step()
        if settings["training_mode"] == "joint":
            self.mtp_grad_norm = self._mtp_gradient_norm()
        update = (
            settings["training_mode"] != "mtp_only"
            or int(self.mtp_global_valid_tokens) > 0
        )
        grad_norm = super().optimizer_step(update=update)
        if settings["training_mode"] == "mtp_only":
            self.mtp_grad_norm = grad_norm.detach()
        if update:
            self.optimizer_updates += 1
        else:
            self.skipped_empty_mtp_steps += 1
        self._audit_mtp_parameters(optimizer=True)
        return grad_norm

    @torch.no_grad()
    def _mtp_gradient_norm(self):
        mesh = self.parallel_dims.get_mesh("tp") if self.parallel_dims.tp > 1 else None
        sharded, replicated, seen = [], [], set()
        for model in self.model_parts:
            for module in model.mtp_layers.modules():
                for name, parameter in module.named_parameters(recurse=False):
                    if parameter.grad is None or id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    placement = (
                        resolve_placements(
                            module._sharding_config.state_shardings[name], mesh
                        )[0]
                        if mesh is not None
                        else None
                    )
                    (
                        sharded
                        if placement is None or placement.is_shard()
                        else replicated
                    ).append(parameter.grad)
        squared = torch.zeros((), device=self.device)
        for gradients, divisor in (
            (sharded, 1),
            (replicated, mesh.size() if mesh is not None else 1),
        ):
            if gradients:
                norm = torch.nn.utils.get_total_norm(gradients, foreach=True)
                if isinstance(norm, DTensor):
                    norm = norm.full_tensor()
                squared += norm.square() / divisor
        if mesh is not None:
            dist.all_reduce(squared, group=mesh.get_group())
        return squared.sqrt()

    def state_dict(self):
        state = super().state_dict()
        if self.mtp_settings:
            state.update(
                optimizer_updates=self.optimizer_updates,
                skipped_empty_mtp_steps=self.skipped_empty_mtp_steps,
            )
        return state

    def load_state_dict(self, state):
        if self.mtp_settings:
            for key in ("optimizer_updates", "skipped_empty_mtp_steps"):
                if key not in state:
                    raise ValueError(f"MTP training state is missing {key}")
                value = state[key]
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid MTP training counter {key}={value}")
            if (
                type(state.get("step")) is not int
                or state["optimizer_updates"] + state["skipped_empty_mtp_steps"]
                > state["step"]
            ):
                raise ValueError("MTP optimizer/skip counters exceed consumed steps")
            for key in ("optimizer_updates", "skipped_empty_mtp_steps"):
                setattr(self, key, state[key])
        super().load_state_dict(state)

    @torch.no_grad()
    def clip_grad_norm(self):
        if self.parallel_dims.tp == 1:
            if self.mtp_settings and self.mtp_settings["training_mode"] == "mtp_only":
                parameters = [
                    p
                    for model in self.model_parts
                    for p in model.parameters()
                    if p.requires_grad
                ]
                return dist_utils.clip_grad_norm_(
                    parameters,
                    self.config.training.max_norm,
                    foreach=True,
                    pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                    ep_enabled=self.parallel_dims.ep_enabled,
                )
            return super().clip_grad_norm()
        mesh = self.parallel_dims.get_mesh("tp")
        sharded, replicated, seen = [], [], set()
        for model in self.model_parts:
            for module in model.modules():
                config = getattr(module, "_sharding_config", None)
                for name, param in module.named_parameters(recurse=False):
                    if (
                        not param.requires_grad
                        or param.grad is None
                        or id(param) in seen
                    ):
                        continue
                    seen.add(id(param))
                    if config is None or name not in config.state_shardings:
                        raise ValueError(
                            f"Missing TP placement for {type(module).__name__}.{name}"
                        )
                    placement = resolve_placements(config.state_shardings[name], mesh)[
                        0
                    ]
                    (sharded if placement.is_shard() else replicated).append(param)
        squared = torch.zeros((), device=self.device)
        for parameters, divisor in ((sharded, 1), (replicated, mesh.size())):
            if parameters:
                norm = torch.nn.utils.get_total_norm(
                    [p.grad for p in parameters], foreach=True
                )
                if isinstance(norm, DTensor):
                    norm = norm.full_tensor()
                squared += norm.square() / divisor
        dist.all_reduce(squared, group=mesh.get_group())
        norm = squared.sqrt()
        torch.nn.utils.clip_grads_with_norm_(
            sharded + replicated, self.config.training.max_norm, norm, foreach=True
        )
        return norm


class LoopTrainer(Trainer):
    engine_cls = LoopTrainingEngine

    def prepare_step_microbatches(self, microbatch_groups):
        if not self.engine.mtp_settings:
            return super().prepare_step_microbatches(microbatch_groups)
        from torchtitan.experiments.loop_qwen35.mtp_data import (
            align_mtp_batch,
            MTPMicrobatch,
        )

        # All prefetched batches are still complete and CP/TP-replicated.
        # Only DP owns distinct samples and therefore participates in SUM.
        local_count = sum(
            int(align_mtp_batch(microbatch.as_input_dict())["mtp_valid_tokens"])
            for group in microbatch_groups
            for microbatch in group
        )
        count = torch.tensor(local_count, dtype=torch.int64, device=self.engine.device)
        dims = self.engine.parallel_dims
        if dims.dp_enabled:
            count = dist_utils.dist_sum_tensor(count, dims.get_mesh("dp"))
        self.engine.mtp_global_valid_tokens = count
        return [
            [
                MTPMicrobatch(source=microbatch, mtp_global_valid_tokens=count)
                for microbatch in group
            ]
            for group in microbatch_groups
        ]

    def log_step_metrics(
        self,
        *,
        accumulated_loss,
        local_valid_tokens,
        global_valid_tokens,
        grad_norm,
        lr_metrics,
    ):
        if not self.engine.mtp_settings:
            return super().log_step_metrics(
                accumulated_loss=accumulated_loss,
                local_valid_tokens=local_valid_tokens,
                global_valid_tokens=global_valid_tokens,
                grad_norm=grad_norm,
                lr_metrics=lr_metrics,
            )
        report = self.engine.mtp_step_statistics()
        report.update(lr_metrics)
        report["grad_norm/trainable"] = float(grad_norm)
        report["mtp/grad_norm"] = float(self.engine.mtp_grad_norm)
        self.engine.last_mtp_step_metrics = report
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(
                "Loop MTP step metrics: %s",
                json.dumps(report, sort_keys=True, allow_nan=False),
            )
        # TensorBoard accepts only scalars; mode/role remain explicit in JSON.
        numeric_metrics = {
            key: value
            for key, value in report.items()
            if isinstance(value, (int, float))
        }
        self.metrics_processor.log(
            self.engine.num_completed_steps,
            report["loss/train"],
            report["loss/max_local_train"],
            float(grad_norm),
            extra_metrics=numeric_metrics,
        )
