# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Same-topology rank checkpoints for Loop's local-TP FSDP adapter.

Native DCP cannot infer the TP placement of this adapter's plain local tensors.
Keep rank files explicitly separate, including optimizer, data cursor and RNG.
"""

import copy
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.config import Configurable

logger = logging.getLogger(__name__)

MTP_CHECKPOINT_SCHEMA = "torchtitan.experiments.loop_qwen35.loop_mtp"
MTP_CHECKPOINT_VERSION = 1
MTP_CONTRACT_FIELDS = (
    "num_layers",
    "training_mode",
    "loss_scaling_factor",
    "prediction_offset",
    "teacher_forcing",
    "normalization",
    "hidden_stage",
    "detach_backbone",
    "automatic_unfreezing",
    "weight_format",
    "weight_format_version",
)


def mtp_checkpoint_metadata(model, resolved, *, require_source=True):
    """Describe the algorithm and canonical parameter sets, independent of AC."""
    from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import parameter_manifest

    settings = getattr(model, "mtp_settings", None)
    metadata = {
        "schema": MTP_CHECKPOINT_SCHEMA,
        "version": MTP_CHECKPOINT_VERSION,
        "enabled": bool(settings and settings["enabled"]),
    }
    if not metadata["enabled"]:
        return metadata
    configuration = (resolved or {}).get("mtp") or settings
    missing = set(MTP_CONTRACT_FIELDS) - configuration.keys()
    if missing:
        raise ValueError(f"MTP checkpoint configuration is missing {sorted(missing)}")
    metadata.update(
        {key: copy.deepcopy(configuration[key]) for key in MTP_CONTRACT_FIELDS}
    )
    metadata.update(parameter_manifest(model))
    if require_source:
        if "source_manifest" not in configuration:
            raise ValueError(
                "MTP checkpoint requires the initial tensor source manifest"
            )
        metadata["source_manifest"] = copy.deepcopy(configuration["source_manifest"])
    return metadata


def normalized_mtp_metadata(metadata):
    """Legacy checkpoints are disabled; an enabled contract must be complete."""
    from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import MTP_HF_TO_NATIVE

    value = metadata.get("mtp")
    if value is None:
        if (metadata.get("resolved_config") or {}).get("mtp", {}).get("enabled"):
            raise ValueError("Enabled MTP checkpoint is missing its schema metadata")
        return {"enabled": False}
    if (
        value.get("schema") != MTP_CHECKPOINT_SCHEMA
        or value.get("version") != MTP_CHECKPOINT_VERSION
    ):
        raise ValueError("Unsupported MTP checkpoint schema/version")
    if type(value.get("enabled")) is not bool:
        raise ValueError("MTP checkpoint must declare enabled as a boolean")
    if not value["enabled"]:
        return {"enabled": False}
    required = {*MTP_CONTRACT_FIELDS, "source_manifest"}
    required.update(
        f"{kind}_{suffix}"
        for kind in ("trainable", "frozen")
        for suffix in ("parameters", "parameters_sha256", "numel")
    )
    missing = required - value.keys()
    if missing:
        raise ValueError(f"MTP checkpoint metadata is missing {sorted(missing)}")
    for kind in ("trainable", "frozen"):
        names = value[f"{kind}_parameters"]
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError(f"Invalid MTP {kind} parameter names")
        if names != sorted(set(names)):
            raise ValueError(f"MTP {kind} parameter names must be unique and sorted")
        digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
        if digest != value[f"{kind}_parameters_sha256"]:
            raise ValueError(f"MTP {kind} parameter-set hash mismatch")
        if type(value[f"{kind}_numel"]) is not int or value[f"{kind}_numel"] < 0:
            raise ValueError(f"Invalid MTP {kind} parameter count")
    if set(value["trainable_parameters"]) & set(value["frozen_parameters"]):
        raise ValueError("MTP trainable and frozen parameter sets overlap")
    source = value["source_manifest"]
    tensors = source.get("tensors", {})
    if set(tensors) != set(MTP_HF_TO_NATIVE):
        raise ValueError("MTP checkpoint source manifest must contain all 15 tensors")
    for name, tensor in tensors.items():
        if tensor.get("target") != MTP_HF_TO_NATIVE[name] or not tensor.get("finite"):
            raise ValueError(f"Invalid MTP source manifest entry: {name}")
        digest = tensor.get("sha256", "")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Invalid MTP source tensor hash: {name}")
    return copy.deepcopy(value)


def validate_mtp_resume_metadata(metadata, model, resolved):
    """Compare algorithm and parameter sets before building an optimizer."""
    saved = normalized_mtp_metadata(metadata)
    current = mtp_checkpoint_metadata(model, resolved, require_source=False)
    if saved["enabled"] != current["enabled"]:
        raise ValueError(
            "Native resume cannot enable or disable MTP; use the same training mode"
        )
    if not saved["enabled"]:
        return saved
    for key in (
        *MTP_CONTRACT_FIELDS,
        "trainable_parameters",
        "trainable_parameters_sha256",
        "trainable_numel",
        "frozen_parameters",
        "frozen_parameters_sha256",
        "frozen_numel",
    ):
        if saved[key] != current[key]:
            raise ValueError(
                f"Native MTP resume requires identical {key}; stage transitions are not full resume"
            )
    return saved


def optimizer_parameter_groups(optimizers):
    """Record ordered optimizer membership without materializing its state."""
    from torchtitan.components.checkpointer.utils import canonical_fqn

    groups = []
    for optimizer in optimizers:
        parameters = []
        for group in optimizer.param_groups:
            if "param_names" not in group or len(group["param_names"]) != len(
                group["params"]
            ):
                raise ValueError(
                    "MTP native checkpoints require named optimizer parameter groups"
                )
            parameters.append([canonical_fqn(name) for name in group["param_names"]])
        groups.append({"optimizer": type(optimizer).__name__, "parameters": parameters})
    return groups


def validate_optimizer_membership(groups, trainable_names):
    names = [
        name
        for optimizer in groups
        for group in optimizer["parameters"]
        for name in group
    ]
    if len(names) != len(set(names)) or sorted(names) != trainable_names:
        raise ValueError(
            "MTP checkpoint optimizer must contain exactly its trainable parameter set"
        )


def select_native_checkpoint(folder, *, resume, step=-1, resume_from=""):
    folder = Path(folder)
    if step >= 0:
        selected = folder / f"step-{step}"
        if not (selected / "complete.json").is_file():
            raise ValueError(f"Native checkpoint is incomplete or missing: {selected}")
        return selected
    if not resume:
        return None
    candidates = [
        path for path in folder.glob("step-*") if (path / "complete.json").is_file()
    ]
    if candidates:
        return max(candidates, key=lambda path: int(path.name.split("-")[1]))
    if resume_from:
        selected = Path(resume_from)
        if not (selected / "complete.json").is_file():
            raise ValueError(f"Native checkpoint is incomplete or missing: {selected}")
        return selected
    return None


def checkpoint_topology(model, dims):
    return {
        "world_size": dist.get_world_size(),
        "tp": dims.tp,
        "cp": dims.cp,
        "pp": dims.pp,
        "dp_replicate": dims.dp_replicate,
        "dp_shard": dims.dp_shard,
        "loop": model.loop_settings.to_dict(),
        "mhc": getattr(getattr(model, "config", None), "mhc", None),
    }


def validate_native_metadata(metadata, model, dims, resolved):
    metadata = copy.deepcopy(metadata)
    saved_topology = dict(metadata["topology"])
    saved_topology.setdefault("mhc", None)
    saved_topology.setdefault("dp_replicate", 1)
    saved_topology.setdefault(
        "dp_shard",
        saved_topology["world_size"]
        // (saved_topology["tp"] * saved_topology["cp"] * saved_topology["pp"]),
    )
    if saved_topology != checkpoint_topology(model, dims):
        raise ValueError(
            "Rank checkpoint resume requires identical world size, TP/CP/PP, DP layout and Loop configuration"
        )
    saved_config = metadata.get("resolved_config")
    if resolved is not None:
        if saved_config is None:
            if resolved.get("mtp", {}).get("enabled"):
                raise ValueError(
                    "Native MTP checkpoint is missing base architecture metadata"
                )
            logger.warning(
                "Legacy checkpoint has no base-model metadata; relying on strict tensor-shape checks"
            )
        elif saved_config["base_model"] != resolved["base_model"]:
            raise ValueError(
                "Native checkpoint resume requires the same base model architecture"
            )
    return validate_mtp_resume_metadata(metadata, model, resolved)


def validate_mtp_optimizer_config(mtp, model, engine_config):
    from torchtitan.components.optimizer import OptimizersContainer

    for key in (
        "optimizer_groups",
        "optimizer_keys",
        "model_keys",
        "optimizer_config",
        "scheduler_config",
        "effective_schedule_steps",
    ):
        if key not in mtp:
            raise ValueError(f"MTP checkpoint metadata is missing {key}")
    validate_optimizer_membership(mtp["optimizer_groups"], mtp["trainable_parameters"])
    current = json.loads(json.dumps(engine_config.optimizer.to_dict()))
    if current != mtp["optimizer_config"]:
        raise ValueError("Native MTP resume requires identical optimizer configuration")
    groups, _ = OptimizersContainer._build_param_groups(
        model, engine_config.optimizer.param_groups, {}
    )
    expected = [
        {"optimizer": name, "parameters": [group["param_names"] for group in values]}
        for name, values in groups.items()
    ]
    if expected != mtp["optimizer_groups"]:
        raise ValueError("Native MTP resume optimizer parameter groups differ")
    schedule = engine_config.lr_scheduler.to_dict()
    saved_schedule = dict(mtp["scheduler_config"])
    requested_steps = schedule.pop("total_steps")
    saved_schedule.pop("total_steps")
    if schedule != saved_schedule or (
        requested_steps is not None
        and requested_steps != mtp["effective_schedule_steps"]
    ):
        raise ValueError(
            "Native MTP resume requires the saved scheduler configuration and horizon"
        )
    horizon = mtp["effective_schedule_steps"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Invalid MTP checkpoint scheduler horizon")
    # Extending the consumed-step stopping point must not redefine the LR curve.
    engine_config.lr_scheduler.total_steps = horizon


def preflight_native_resume(engine):
    """Validate native contracts after freezing and before optimizer construction."""
    config = engine.config.checkpointer
    if not isinstance(config, LoopCheckpoint.Config):
        return
    selected = select_native_checkpoint(
        Path(engine.output_dir) / "checkpoints",
        resume=config.resume,
        step=config.load_step,
        resume_from=config.resume_from,
    )
    if selected is None:
        return
    metadata = json.loads((selected / "complete.json").read_text())
    model = engine.model_parts[0]
    model.audit_mtp_trainability()
    mtp = validate_native_metadata(
        metadata, model, engine.parallel_dims, config.resolved_config
    )
    if not (selected / f"rank-{dist.get_rank()}.pt").is_file():
        raise ValueError(
            f"Native checkpoint is missing the current rank payload: {selected}"
        )
    if mtp["enabled"]:
        validate_mtp_optimizer_config(mtp, model, engine.config)
        config.resolved_config["mtp"]["source_manifest"] = copy.deepcopy(
            mtp["source_manifest"]
        )
    engine._native_resume_preflight = (selected, metadata)


def validate_mtp_payload(state, metadata, *, parts, optimizers, schedulers):
    """Reject incomplete state before assigning weights or optimizer tensors."""
    mtp = normalized_mtp_metadata(metadata)
    if not mtp["enabled"]:
        return
    required = {
        "model",
        "optimizer",
        "scheduler",
        "mtp_scheduler_states",
        "dataloader",
        "train_state",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "mtp",
    }
    if required - state.keys():
        raise ValueError(
            f"MTP rank checkpoint is missing {sorted(required - state.keys())}"
        )
    if state["mtp"] != metadata["mtp"]:
        raise ValueError("MTP rank metadata differs from the completion marker")
    if (
        len(state["model"]) != len(parts)
        or [sorted(value) for value in state["model"]] != mtp["model_keys"]
    ):
        raise ValueError("MTP checkpoint model tensor keys are incomplete")
    for part, saved in zip(parts, state["model"], strict=True):
        current = part.state_dict()
        if current.keys() != saved.keys():
            raise ValueError(
                "MTP checkpoint does not contain the complete backbone and MTP tensors"
            )
        for name, destination in current.items():
            if (
                destination.shape != saved[name].shape
                or destination.dtype != saved[name].dtype
            ):
                raise ValueError(f"MTP checkpoint tensor shape/dtype mismatch: {name}")
    if optimizer_parameter_groups(optimizers) != mtp["optimizer_groups"]:
        raise ValueError("MTP checkpoint optimizer parameter groups differ")
    if sorted(state["optimizer"]) != mtp["optimizer_keys"]:
        raise ValueError("MTP checkpoint optimizer state is incomplete")
    expected = optimizers.state_dict()
    if expected.keys() != state["optimizer"].keys():
        raise ValueError(
            "MTP checkpoint optimizer state keys do not match the current optimizer"
        )
    for name, destination in expected.items():
        value = state["optimizer"][name]
        if isinstance(destination, torch.Tensor):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != destination.shape
                or value.dtype != destination.dtype
            ):
                raise ValueError(
                    f"MTP optimizer state tensor shape/dtype mismatch: {name}"
                )
    full_schedulers = state["mtp_scheduler_states"]
    if len(full_schedulers) != len(schedulers):
        raise ValueError("MTP checkpoint scheduler state is incomplete")
    for scheduler, saved in zip(schedulers, full_schedulers, strict=True):
        if scheduler.state_dict().keys() != saved.keys():
            raise ValueError("MTP checkpoint scheduler fields are incomplete")
    train = state["train_state"]
    for key in ("step", "ntokens_seen", "optimizer_updates", "skipped_empty_mtp_steps"):
        if type(train.get(key)) is not int or train[key] < 0:
            raise ValueError(
                f"MTP checkpoint is missing a valid training counter: {key}"
            )
    if (
        train["step"] != metadata["step"]
        or train["optimizer_updates"] + train["skipped_empty_mtp_steps"] > train["step"]
    ):
        raise ValueError("MTP checkpoint counters disagree with consumed steps")
    if state["scheduler"].get("last_epoch") != train["optimizer_updates"]:
        raise ValueError("MTP scheduler epoch must equal actual optimizer updates")
    if any(
        saved.get("last_epoch") != train["optimizer_updates"]
        for saved in full_schedulers
    ):
        raise ValueError(
            "MTP full scheduler state disagrees with actual optimizer updates"
        )


class LoopCheckpoint(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        initial_load_path: str = ""
        initial_step: int = 0
        initial_rank_weights: str = ""
        resolved_config: dict | None = None
        interval: int = 100
        load_step: int = -1
        resume: bool = True
        resume_from: str = ""
        load_only: bool = False
        strict_initial_weights: bool = False

    def __init__(
        self,
        config,
        *,
        model_parts,
        optimizers,
        lr_schedulers,
        dataloader,
        states,
        sd_adapter,
        base_folder,
        **kwargs,
    ):
        self.config = config
        self.parts = model_parts
        self.optimizers, self.schedulers = optimizers, lr_schedulers
        self.dataloader, self.engine = dataloader, states["train_state"]
        self.adapter = sd_adapter
        self.folder = Path(base_folder) / "checkpoints"
        self.topology = {
            "world_size": dist.get_world_size(),
            "tp": self.engine.parallel_dims.tp,
            "cp": self.engine.parallel_dims.cp,
            "pp": self.engine.parallel_dims.pp,
            "dp_replicate": self.engine.parallel_dims.dp_replicate,
            "dp_shard": self.engine.parallel_dims.dp_shard,
            "loop": self.parts[0].loop_settings.to_dict(),
            "mhc": self.parts[0].config.mhc,
        }

    @torch.no_grad()
    def _load_hf(self):
        source = Path(self.config.initial_load_path)
        resolved = self.config.resolved_config
        mtp_enabled = bool(resolved and resolved.get("mtp", {}).get("enabled"))
        if mtp_enabled:
            from torchtitan.experiments.loop_qwen35.model_config import (
                validate_mtp_source_config,
            )

            validate_mtp_source_config(
                json.loads((source / "config.json").read_text()),
                resolved["mtp"],
                model_flavor=resolved["model_flavor"],
            )
        index_path = source / "model.safetensors.index.json"
        filenames = (
            sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
            if index_path.is_file()
            else ["model.safetensors"]
        )
        weights = {}
        for filename in filenames:
            weights.update(load_file(str(source / filename), device="cpu"))
        # A reduced-depth regression retains HF layer indices and dimensions.
        # Filter before the adapter consults per-layer attention/GDN configs.
        layer_prefix = f"{self.adapter.hf_language_model_prefix}.layers."
        retained_layers = len(self.parts[0].config.layers)
        weights = {
            key: value
            for key, value in weights.items()
            if not (
                key.startswith(layer_prefix)
                and int(key[len(layer_prefix) :].split(".", 1)[0]) >= retained_layers
            )
        }
        # Preserve an independent HF output head (9B). For tied checkpoints
        # without that tensor (4B), initialize it from the trained embedding.
        embedding_key = f"{self.adapter.hf_language_model_prefix}.embed_tokens.weight"
        if "lm_head.weight" not in weights:
            if (
                self.config.resolved_config
                and not self.config.resolved_config["base_model"]["tie_word_embeddings"]
            ):
                raise KeyError("Untied HF checkpoint is missing lm_head.weight")
            weights["lm_head.weight"] = weights[embedding_key]
        full = self.adapter.from_hf(weights)
        del weights
        if mtp_enabled:
            from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import (
                mtp_source_manifest,
                slice_mtp_for_tp,
            )

            resolved["mtp"]["source_manifest"] = mtp_source_manifest(
                source, self.adapter.mtp_tensor_manifest
            )
        dims = self.engine.parallel_dims
        tp_rank = dims.get_mesh("tp").get_local_rank() if dims.tp > 1 else 0
        loaded = 0
        initialized = []
        from torchtitan.components.checkpointer.utils import canonical_fqn
        from torchtitan.experiments.loop_qwen35.mhc.model import MHCRouter

        allowed_new = {
            canonical_fqn(f"{module_name}.{parameter_name}")
            for part in self.parts
            for module_name, module in part.named_modules()
            if isinstance(module, MHCRouter)
            for parameter_name, _ in module.named_parameters(recurse=False)
        }
        for part in self.parts:
            target = part.state_dict()
            for name, destination in target.items():
                if name not in full:
                    if self.config.strict_initial_weights:
                        raise KeyError(
                            f"Initial checkpoint is missing required model tensor {name}"
                        )
                    if name in allowed_new:
                        initialized.append(name)
                        continue
                    if name == "loop_gate.logits":
                        logger.info(
                            "HF checkpoint has no loop gate; retaining configured initialization"
                        )
                        continue
                    raise KeyError(
                        f"HF checkpoint is missing required model tensor {name}"
                    )
                value = full[name]
                is_mtp = mtp_enabled and name.startswith("mtp_layers.")
                if is_mtp and dims.tp > 1:
                    value = slice_mtp_for_tp(part, name, value, dims.get_mesh("tp"))
                differing = [
                    i
                    for i, (a, b) in enumerate(zip(value.shape, destination.shape))
                    if a != b
                ]
                if value.ndim != destination.ndim:
                    raise ValueError(f"HF tensor rank mismatch for {name}")
                if differing:
                    if is_mtp:
                        raise ValueError(
                            f"Declared MTP partition shape mismatch for {name}: "
                            f"{value.shape} -> {destination.shape}"
                        )
                    if (
                        len(differing) != 1
                        or value.shape[differing[0]]
                        != destination.shape[differing[0]] * dims.tp
                    ):
                        raise ValueError(
                            f"Unsupported HF/TP shape for {name}: {value.shape} -> {destination.shape}"
                        )
                    value = value.chunk(dims.tp, dim=differing[0])[tp_rank]
                value = value.to(
                    device=self.engine.device, dtype=destination.dtype
                ).contiguous()
                if isinstance(destination, DTensor):
                    value = distribute_tensor(
                        value, destination.device_mesh, destination.placements
                    )
                target[name] = value
                loaded += 1
            part.load_state_dict(target, strict=True)
            if mtp_enabled:
                from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import (
                    MTP_HF_TO_NATIVE,
                )

                # Audit stored local FSDP shards without materializing the backbone.
                loaded_state = part.state_dict()
                for name in MTP_HF_TO_NATIVE.values():
                    actual, expected = loaded_state[name], target[name]
                    if isinstance(actual, DTensor):
                        actual, expected = actual.to_local(), expected.to_local()
                    if not torch.equal(actual, expected):
                        raise ValueError(
                            f"Loaded MTP tensor differs from its source partition: {name}"
                        )
        logger.info(
            "Loaded %d local model tensors from HF checkpoint %s", loaded, source
        )
        if initialized:
            logger.info("Initialized ONLY new mHC tensors: %s", initialized)

    def _save_resolved_config(self):
        if dist.get_rank() == 0 and self.config.resolved_config is not None:
            resolved = self.config.resolved_config
            self.folder.parent.mkdir(parents=True, exist_ok=True)
            (self.folder.parent / "resolved_config.json").write_text(
                json.dumps(resolved, indent=2) + "\n"
            )
            (self.folder.parent / "loop_config.json").write_text(
                json.dumps(resolved["loop_settings"], indent=2) + "\n"
            )

    @torch.no_grad()
    def _load_rank_weights(self):
        """Import weights only when CP changes preserve the flattened FSDP mesh."""
        source = Path(self.config.initial_rank_weights)
        metadata = json.loads((source / "complete.json").read_text())
        saved = metadata["topology"]
        current = self.topology
        for key in ("world_size", "tp", "pp", "dp_replicate", "loop"):
            if saved.get(key) != current[key]:
                raise ValueError(f"Weights-only rank import requires identical {key}")
        if saved["dp_shard"] * saved["cp"] != current["dp_shard"] * current["cp"]:
            raise ValueError(
                "Weights-only rank import requires identical flattened FSDP size"
            )
        resolved = self.config.resolved_config
        if (
            resolved is None
            or metadata.get("resolved_config", {}).get("base_model")
            != resolved["base_model"]
        ):
            raise ValueError(
                "Weights-only rank import requires matching base-model metadata"
            )
        # mmap avoids eagerly reading the unused optimizer payload into memory.
        state = torch.load(
            source / f"rank-{dist.get_rank()}.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for part, weights in zip(self.parts, state["model"], strict=True):
            target = part.state_dict()
            if target.keys() != weights.keys():
                raise ValueError(
                    "Weights-only rank import requires identical model keys"
                )
            for name, destination in target.items():
                value = weights[name]
                if destination.shape != value.shape or destination.dtype != value.dtype:
                    raise ValueError(f"Weights-only shape/dtype mismatch: {name}")
                if isinstance(destination, DTensor) != isinstance(value, DTensor):
                    raise ValueError(f"Weights-only tensor layout mismatch: {name}")
                if isinstance(destination, DTensor):
                    if destination.placements != value.placements or not torch.equal(
                        destination.device_mesh.mesh.cpu(), value.device_mesh.mesh.cpu()
                    ):
                        raise ValueError(
                            f"Weights-only FSDP mesh/placement mismatch: {name}"
                        )
                    destination, value = destination.to_local(), value.to_local()
                if destination.shape != value.shape:
                    raise ValueError(f"Weights-only local shard mismatch: {name}")
                destination.copy_(value)
            part.load_state_dict(target, strict=True)
        logger.info(
            "Imported MODEL WEIGHTS ONLY from %s (source step %s); optimizer, scheduler, "
            "data and RNG start fresh; initial step %s",
            source,
            metadata["step"],
            self.config.initial_step,
        )

    def load(self, step=-1):
        candidates = []
        if self.config.resume and self.folder.exists():
            candidates = [
                p for p in self.folder.glob("step-*") if (p / "complete.json").is_file()
            ]
        selected = (
            max(candidates, key=lambda p: int(p.name.split("-")[1]))
            if candidates
            else None
        )
        if selected is None and self.config.resume and self.config.resume_from:
            selected = Path(self.config.resume_from)
        if step >= 0:
            selected = self.folder / f"step-{step}"
        if selected is None:
            if self.config.initial_rank_weights:
                self._load_rank_weights()
            else:
                self._load_hf()
            self.engine.num_completed_steps = self.config.initial_step
            logger.info(
                "Initial weight import: next step %d; optimizer, scheduler, data and RNG start fresh",
                self.config.initial_step + 1,
            )
            self._save_resolved_config()
            return False
        metadata = json.loads((selected / "complete.json").read_text())
        metadata = copy.deepcopy(metadata)
        self.validate_extra_metadata(metadata, selected)
        mtp = validate_mtp_resume_metadata(
            metadata, self.parts[0], self.config.resolved_config
        )
        if mtp["enabled"]:
            validate_mtp_optimizer_config(mtp, self.parts[0], self.engine.config)
        current_config = copy.deepcopy(
            {
                "topology": self.topology,
                "resolved_config": self.config.resolved_config or {},
            }
        )
        saved_experiment = copy.deepcopy(
            metadata.get("resolved_config", {}).get("experiment")
        )
        current_experiment = copy.deepcopy(
            current_config["resolved_config"].get("experiment")
        )
        if (
            mtp["enabled"]
            and saved_experiment is not None
            and current_experiment is not None
        ):
            saved_experiment.pop("train_iters", None)
            current_experiment.pop("train_iters", None)
        if saved_experiment != current_experiment:
            raise ValueError(
                "Resume requires identical mHC experiment and learning-rate settings"
            )
        if metadata["topology"] != self.topology:
            raise ValueError(
                "Checkpoint resume requires identical parallel topology, Loop and mHC settings"
            )
        resolved = self.config.resolved_config
        saved_config = metadata.get("resolved_config")
        if resolved is not None:
            if saved_config is None:
                logger.warning(
                    "Legacy checkpoint has no base-model metadata; relying on strict tensor-shape checks"
                )
            elif saved_config["base_model"] != resolved["base_model"]:
                raise ValueError(
                    "Native checkpoint resume requires the same base model architecture"
                )
        state = torch.load(
            selected / f"rank-{dist.get_rank()}.pt",
            map_location=self.engine.device,
            weights_only=False,
        )
        validate_mtp_payload(
            state,
            metadata,
            parts=self.parts,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
        )
        for part, saved in zip(self.parts, state["model"], strict=True):
            part.load_state_dict(saved, strict=True)
        self.optimizers.load_state_dict(state["optimizer"])
        if mtp["enabled"]:
            for scheduler, saved in zip(
                self.schedulers, state["mtp_scheduler_states"], strict=True
            ):
                scheduler.load_state_dict(saved)
                for group, lr in zip(
                    scheduler.optimizer.param_groups,
                    scheduler.get_last_lr(),
                    strict=True,
                ):
                    group["lr"] = lr
            resolved["mtp"]["source_manifest"] = copy.deepcopy(mtp["source_manifest"])
        else:
            self.schedulers.load_state_dict(state["scheduler"])
        self.dataloader.load_state_dict(state["dataloader"])
        self.engine.load_state_dict(state["train_state"])
        self.load_extra_state(state, metadata, selected)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        logger.info("Resumed all training state from %s", selected)
        self._save_resolved_config()
        return True

    def validate_extra_metadata(self, metadata, source):
        pass

    def load_extra_state(self, state, metadata, source):
        pass

    def extra_metadata(self):
        return {}

    def save(self, step, *, last_step=False):
        if self.config.load_only or (not last_step and step % self.config.interval):
            return False
        extras = self.extra_metadata()
        mtp = mtp_checkpoint_metadata(self.parts[0], self.config.resolved_config)
        if mtp["enabled"]:
            self.engine._audit_mtp_parameters(optimizer=True)
            mtp["optimizer_groups"] = optimizer_parameter_groups(self.optimizers)
            validate_optimizer_membership(
                mtp["optimizer_groups"], mtp["trainable_parameters"]
            )
            mtp["optimizer_config"] = json.loads(
                json.dumps(self.engine.config.optimizer.to_dict())
            )
            mtp["scheduler_config"] = self.engine.config.lr_scheduler.to_dict()
            mtp["effective_schedule_steps"] = (
                self.engine.config.lr_scheduler.total_steps
                or self.engine.config.training.steps
            )
            normalized_mtp_metadata({"mtp": mtp})
        destination = self.folder / f"step-{step}"
        if dist.get_rank() == 0:
            destination.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        state = {
            "model": [p.state_dict() for p in self.parts],
            "optimizer": self.optimizers.state_dict(),
            "scheduler": self.schedulers.state_dict(),
            "dataloader": self.dataloader.state_dict(),
            "train_state": self.engine.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        if mtp["enabled"]:
            state["mtp_scheduler_states"] = [
                scheduler.state_dict() for scheduler in self.schedulers
            ]
            mtp["optimizer_keys"] = sorted(state["optimizer"])
            mtp["model_keys"] = [sorted(value) for value in state["model"]]
            state["mtp"] = copy.deepcopy(mtp)
        temporary = destination / f"rank-{dist.get_rank()}.pt.tmp"
        torch.save(state, temporary)
        temporary.rename(destination / f"rank-{dist.get_rank()}.pt")
        dist.barrier()
        if dist.get_rank() == 0:
            marker = destination / "complete.json.tmp"
            marker.write_text(
                json.dumps(
                    {
                        "topology": self.topology,
                        "step": step,
                        "resolved_config": self.config.resolved_config,
                        **extras,
                        "mtp": mtp,
                    }
                )
            )
            marker.rename(destination / "complete.json")
        dist.barrier()
        logger.info(
            "Saved model, optimizer, scheduler, data cursor and RNG: %s", destination
        )
        return True

    def maybe_wait_for_staging(self):
        pass

    def close(self):
        pass
