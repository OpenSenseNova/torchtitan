# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan Qwen3.5 Loop model.

This module keeps the native TorchTitan Qwen3.5 implementation for attention,
GatedDeltaNet, tensor parallel sharding, and checkpoint conversion.  It adds
a Loop schedule: the layer interval
``[loop_start, loop_end)`` is executed repeatedly and an FP32 channel gate
mixes the repeated candidates.

The file is intentionally self-contained so it can be loaded by a TorchTitan
recipe without importing the Transformers Loop implementation.  Its CP path
keeps token/vision scattering replicated, exchanges token/head shards around
GatedDeltaNet with all-to-all, and defaults to head-balanced Ulysses attention.
"""

from __future__ import annotations

import dataclasses
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from typing import Any

import spmd_types as spmd
import torch
from spmd_types import SpmdType
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    set_current_spmd_mesh,
    spmd_local_context,
)

from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import Qwen35MTPBlock
from torchtitan.models.common.attention import FlexInnerAttention, VarlenInnerAttention
from torchtitan.models.common.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
    KVAllGatherCPVarlenInnerAttention,
    make_cp_varlen_metadata,
    UlyssesCPFlexInnerAttention,
    UlyssesCPVarlenInnerAttention,
)
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    decoder_input_sharding,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.common.multimodal import (
    build_vision_bank_indices,
    gather_vision_embeds,
    get_vision_positions,
)
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.qwen3_5 import model_registry as qwen35_model_registry
from torchtitan.models.qwen3_5.gdn import (
    AllToAllCPInnerGatedDeltaNet,
    InnerGatedDeltaNet,
)
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.models.qwen3_5.parallelize import parallelize_qwen3_5
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import Module, ModuleList
from torchtitan.protocols.sharding import ShardingConfig


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


@dataclass(kw_only=True, slots=True)
class LoopSettings:
    """Serializable Loop schedule and gate settings."""

    start: int = 8
    end: int = 16
    counts: list[int] = field(default_factory=lambda: [1, 2])
    probabilities: list[float] = field(default_factory=lambda: [0.25, 0.75])
    eval_count: int = 2
    gate_init: float = 0.02
    seed: int = 42
    use_gate: bool = True

    def validate(self, num_layers: int) -> None:
        if type(self.use_gate) is not bool:
            raise ValueError("use_gate must be a boolean")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("Loop start and end must be integers")
        if type(self.seed) is not int:
            raise ValueError("Loop seed must be an integer")
        if not 0 <= self.start < self.end <= num_layers:
            raise TypeError("Loop interval must satisfy 0 <= start < end <= num_layers")
        if not self.counts or any(
            type(count) is not int or count < 1 for count in self.counts
        ):
            raise ValueError("counts must contain positive integers")
        if len(self.counts) != len(self.probabilities):
            raise ValueError("counts and probabilities must have equal length")
        if any(
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or probability < 0
            for probability in self.probabilities
        ):
            raise ValueError("probabilities must be finite non-negative numbers")
        if abs(sum(self.probabilities) - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to one")
        if type(self.eval_count) is not int or self.eval_count < 1:
            raise ValueError("eval_count must be a positive integer")
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError("gate_init must be between zero and one")

    def count_for_step(self, step: int) -> int:
        return random.Random(self.seed + int(step)).choices(
            self.counts, self.probabilities
        )[0]

    # Keep the names used by the Transformers Loop callback/configuration.
    def at_step(self, step: int) -> int:
        return self.count_for_step(step)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoopGate(Module):
    """FP32 channel gate used to mix a repeated Loop candidate."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        initial: float
        enabled: bool = True

    def __init__(self, config: Config):
        super().__init__()
        self.enabled = config.enabled
        self.logits = nn.Parameter(torch.empty(config.dim, dtype=torch.float32))

    def forward(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate is None:
            return previous + self.logits.sum().to(previous.dtype) * 0
        if not self.enabled:
            return candidate
        gate = self.logits.float().sigmoid()
        return torch.addcmul(
            previous.float(), gate, candidate.float() - previous.float()
        ).to(previous.dtype)


class LoopInputShard(Module):
    """Shard multimodal embeddings once, before any repeated decoder block.

    The input boundary shards CP; the output boundary shards TP when SP is
    enabled. Each redistribution changes one mesh axis, in CP-then-TP order.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


def _loop_gate_config(dim: int, settings: LoopSettings) -> LoopGate.Config:
    """Build the gate config with replicated TP/CP placement."""
    initial_logit = _logit(settings.gate_init)
    return LoopGate.Config(
        dim=dim,
        initial=settings.gate_init,
        enabled=settings.use_gate,
        param_init={
            "logits": partial(nn.init.constant_, val=initial_logit),
        },
        # The gate is a hidden-size vector.  Replicate it across TP and CP;
        # FSDP handles its data-parallel sharding at the decoder root.
        sharding_config=ShardingConfig(
            state_shardings={"logits": dense_param_placement(tp=spmd.R)}
        ),
    )


def _cp_replicated_token_id() -> SpmdType:
    """Token IDs replicated over CP for multimodal scatter before sharding."""
    return SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.R,
            MeshAxisName.TP: spmd.R,
        },
        partition_spec=spmd.PartitionSpec(MeshAxisName.DP),
    )


def _add_cp_replication(layout):
    """Add a CP=Replicate declaration to a vision layout."""
    if layout is None:
        return None
    if isinstance(layout, tuple):
        return tuple(_add_cp_replication(item) for item in layout)
    if not isinstance(layout, SpmdType):
        return layout
    original = dict(layout.local_type)
    # DeviceMesh slicing requires the same canonical axis order as the dense
    # mesh (DP, CP, TP). Vision configs predate CP and store DP, TP; appending
    # CP would otherwise request the invalid order DP, TP, CP.
    local_type = {}
    for axis in (MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP):
        if axis in original or axis == MeshAxisName.CP:
            local_type[axis] = original.get(axis, spmd.R)
    local_type.update(
        (axis, value) for axis, value in original.items() if axis not in local_type
    )
    return SpmdType(local_type, partition_spec=layout.partition_spec)


def _add_cp_to_sharding(sharding: ShardingConfig | None) -> None:
    if sharding is None:
        return
    sharding.state_shardings = {
        name: _add_cp_replication(layout)
        for name, layout in sharding.state_shardings.items()
    }
    for attr in (
        "in_src_shardings",
        "in_dst_shardings",
        "out_src_shardings",
        "out_dst_shardings",
    ):
        value = getattr(sharding, attr)
        if isinstance(value, dict):
            setattr(
                sharding,
                attr,
                {name: _add_cp_replication(layout) for name, layout in value.items()},
            )
        else:
            setattr(sharding, attr, _add_cp_replication(value))


def _add_cp_to_config_tree(value: Any, seen: set[int] | None = None) -> None:
    """Annotate the vision config tree with CP-replicated layouts."""
    if seen is None:
        seen = set()
    if value is None or id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, ShardingConfig):
        _add_cp_to_sharding(value)
        return
    if dataclasses.is_dataclass(value):
        for config_field in dataclasses.fields(value):
            _add_cp_to_config_tree(getattr(value, config_field.name), seen)
    elif isinstance(value, dict):
        for item in value.values():
            _add_cp_to_config_tree(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _add_cp_to_config_tree(item, seen)


def _cp_sequence_colwise_config() -> ShardingConfig:
    """Head-parallel projection operating on the local CP token shard."""
    config = colwise_config()
    # Sum TP head contributions once in backward without gathering CP tokens.
    config.in_src_shardings = {
        "input": dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    }
    config.in_dst_shardings = {
        "input": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    }
    return config


def _convert_loop_attention_to_cp(config: Qwen35LoopModel.Config) -> None:
    """Reuse native Ulysses attention or the explicit KV-all-gather baseline."""
    ulysses = config.cp_attention_backend == "ulysses"
    for layer in (*config.layers, *getattr(config, "mtp_layers", ())):
        attention = layer.attention
        if attention is None:
            continue
        inner = attention.inner_attention
        if isinstance(inner, VarlenInnerAttention.Config):
            backend = (
                UlyssesCPVarlenInnerAttention
                if ulysses
                else KVAllGatherCPVarlenInnerAttention
            )
            attention.inner_attention = backend.Config(
                window_size=inner.window_size,
                flash_attention_impl=inner.flash_attention_impl,
                **({} if ulysses else {"reduce_dtype": "float32"}),
            )
            continue
        if not isinstance(inner, FlexInnerAttention.Config):
            raise ValueError(
                "Qwen3.5 Loop CP requires Flex or varlen attention; "
                f"got {type(inner).__name__}."
            )
        backend = (
            UlyssesCPFlexInnerAttention if ulysses else KVAllGatherCPFlexInnerAttention
        )
        attention.inner_attention = backend.Config(
            block_size=inner.block_size,
            kernel_options=dict(inner.kernel_options),
            **({} if ulysses else {"reduce_dtype": "float32"}),
        )


def _configure_loop_cp_sharding(config: Qwen35Model.Config, *, enable_sp: bool) -> None:
    """Keep GDN projections token-sharded and exchange heads inside its kernel."""
    _add_cp_to_config_tree(config.vision_encoder)
    token_layout = _cp_replicated_token_id()
    embedding = config.tok_embeddings.sharding_config
    assert embedding is not None
    embedding.in_src_shardings = {"input": token_layout}
    embedding.in_dst_shardings = {"input": token_layout}
    embedding.out_src_shardings = dense_activation_placement(tp=spmd.P, cp=spmd.R)
    embedding.out_dst_shardings = dense_activation_placement(
        tp=spmd.R if enable_sp else spmd.I,
        cp=spmd.R,
    )

    sequence_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    embedding_layout = dense_activation_placement(
        tp=spmd.R if enable_sp else spmd.I,
        cp=spmd.R,
    )
    cp_input_layout = dense_activation_placement(
        tp=spmd.R if enable_sp else spmd.I, cp=spmd.S(0)
    )
    config.input_shard = LoopInputShard.Config(
        sharding_config=ShardingConfig(
            in_src_shardings={"hidden": embedding_layout},
            in_dst_shardings={"hidden": cp_input_layout},
            out_src_shardings=cp_input_layout,
            out_dst_shardings=sequence_layout,
        )
    )

    for layer in config.layers:
        layer_sharding = layer.sharding_config
        assert layer_sharding is not None
        layer_sharding.in_src_shardings = {"x_TD": sequence_layout}
        layer_sharding.in_dst_shardings = {"x_TD": sequence_layout}
        layer_sharding.out_src_shardings = sequence_layout

        if layer.delta_net is None:
            continue
        delta = layer.delta_net

        for name in (
            "in_proj_q",
            "in_proj_k",
            "in_proj_v",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
        ):
            projection = getattr(delta, name)
            # With SP, the GDN boundary gathers tokens and its backward
            # reduce-scatter sums TP head contributions. Adding I -> R at
            # each projection would sum those contributions a second time.
            projection.sharding_config = (
                colwise_config() if enable_sp else _cp_sequence_colwise_config()
            )

        inner = delta.inner_gated_delta_net
        if not isinstance(inner, InnerGatedDeltaNet.Config):
            raise ValueError(
                "Loop GDN all-to-all requires the dense InnerGatedDeltaNet"
            )
        delta.inner_gated_delta_net = AllToAllCPInnerGatedDeltaNet.Config(
            kernel=inner.kernel,
            cp_overlap=config.cp_gdn_overlap,
            param_init=inner.param_init,
            sharding_config=inner.sharding_config,
        )
        delta_sharding = delta.sharding_config
        assert delta_sharding is not None
        # Gather only the TP token shard. CP tokens stay sharded until the
        # existing token/head all-to-all inside the GDN kernel.
        delta_sharding.in_dst_shardings = {"x_TD": cp_input_layout}
        delta_intermediate_layout = dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
        delta_sharding.out_src_shardings = delta_intermediate_layout
        # TP P -> I sums the forward output but keeps its backward invariant.
        # P -> R would sum the same residual gradient again at every GDN block.
        delta_sharding.out_dst_shardings = sequence_layout
        out_projection = delta.out_proj.sharding_config
        assert out_projection is not None
        out_projection.out_src_shardings = dense_activation_placement(
            tp=spmd.P, cp=spmd.S(0)
        )
        out_projection.out_dst_shardings = delta_intermediate_layout


class Qwen35LoopModel(Qwen35Model):
    """Qwen3.5 decoder with a repeated middle layer interval."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        loop: LoopSettings = field(default_factory=LoopSettings)
        loop_gate: LoopGate.Config | None = None
        input_shard: LoopInputShard.Config | None = None
        freeze_vision: bool = False
        mtp: dict[str, Any] | None = None
        mtp_layers: list[Qwen35MTPBlock.Config] = field(default_factory=list)
        mtp_input_shard: LoopInputShard.Config | None = None
        trainable_loop_only: bool = False
        mhc: dict | None = None
        mhc_fused_merge: bool = False
        mhc_fused_backward: bool = True
        mhc_fused_normalization: bool = False
        fsdp_prefetch: bool = False
        cp_attention_backend: str = "ulysses"
        cp_gdn_overlap: bool = True

        def get_nparams_and_flops(self, model, seq_len):
            from torchtitan.experiments.loop_qwen35.metrics import LoopFlops

            model.loop_flops = LoopFlops.from_model(model, self)
            # Runtime metrics replace this single-document estimate with the
            # actual packed lengths and the repeat count selected for each step.
            return (
                model.loop_flops.nparams,
                model.loop_flops.batch([seq_len], self.loop.counts[0]) // seq_len,
            )

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            if self.cp_attention_backend not in ("ulysses", "all_gather"):
                raise ValueError("cp_attention_backend must be ulysses or all_gather")
            if config.parallelism.pipeline_parallel_degree != 1:
                raise ValueError(
                    "Qwen3.5 Loop supports PP=1; pipeline execution is not implemented"
                )
            if config.parallelism.context_parallel_degree > 1:
                if self.cp_attention_backend == "ulysses":
                    tp_cp = (
                        config.parallelism.tensor_parallel_degree
                        * config.parallelism.context_parallel_degree
                    )
                    for layer in (*self.layers, *self.mtp_layers):
                        attention = layer.attention
                        if attention is not None and (
                            attention.n_heads % tp_cp or attention.n_kv_heads % tp_cp
                        ):
                            raise ValueError(
                                f"Ulysses CP requires TP*CP ({tp_cp}) to divide "
                                f"Q heads ({attention.n_heads}) and KV heads ({attention.n_kv_heads}); "
                                "reduce TP/CP or explicitly select --cp-attention-backend all_gather"
                            )
                _convert_loop_attention_to_cp(self)
            super(Qwen35LoopModel.Config, self).update_from_config(
                config=config, **kwargs
            )
            self.loop.validate(len(self.layers))
            if self.loop_gate is None:
                self.loop_gate = _loop_gate_config(self.dim, self.loop)
            if not config.parallelism.enable_sequence_parallel:
                # Residual activations/gradients are TP-invariant without SP.
                # Head-parallel QK/GDN norms retain TP R and sum partial grads.
                self.loop_gate.sharding_config.state_shardings["logits"] = (
                    dense_param_placement(tp=spmd.I)
                )
                for layer in self.layers:
                    layer.attention_norm.sharding_config.state_shardings["weight"] = (
                        dense_param_placement(tp=spmd.I)
                    )
            if config.parallelism.context_parallel_degree > 1:
                tp_cp = (
                    config.parallelism.tensor_parallel_degree
                    * config.parallelism.context_parallel_degree
                )
                for layer in self.layers:
                    delta = layer.delta_net
                    if delta is not None:
                        key_heads = delta.in_proj_q.out_features // delta.key_head_dim
                        value_heads = (
                            delta.in_proj_v.out_features // delta.value_head_dim
                        )
                        if key_heads % tp_cp or value_heads % tp_cp:
                            raise ValueError(
                                f"GDN all-to-all requires TP*CP ({tp_cp}) to divide "
                                f"key heads ({key_heads}) and value heads ({value_heads})"
                            )
                # Ulysses balances work by sharding heads over the complete
                # packed sequence. Keep contiguous token shards for GDN;
                # applying head-tail/PTRR would change its recurrent state.
                config.parallelism.context_parallel_load_balancer = None
                _configure_loop_cp_sharding(
                    self,
                    enable_sp=config.parallelism.enable_sequence_parallel,
                )
            if self.mtp and self.mtp["enabled"]:
                from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import (
                    set_mtp_sharding_config,
                )

                enable_sp = config.parallelism.enable_sequence_parallel
                embedding_layout = self.tok_embeddings.sharding_config.out_dst_shardings
                if config.parallelism.context_parallel_degree > 1:
                    # Slice CP first; the MTP block then restores its TP/SP layout.
                    local_embedding = dense_activation_placement(
                        tp=spmd.R if enable_sp else spmd.I, cp=spmd.S(0)
                    )
                    self.mtp_input_shard = LoopInputShard.Config(
                        sharding_config=ShardingConfig(
                            in_src_shardings={"hidden": embedding_layout},
                            in_dst_shardings={"hidden": local_embedding},
                            out_src_shardings=local_embedding,
                        )
                    )
                    embedding_layout = local_embedding
                for layer in self.mtp_layers:
                    set_mtp_sharding_config(
                        layer, enable_sp=enable_sp, embedding_layout=embedding_layout
                    )

            if self.mhc:
                from torchtitan.experiments.loop_qwen35.mhc.model import MHCBlock

                for index in range(self.mhc["start"], self.mhc["end"]):
                    original = self.layers[index]
                    values = {
                        f.name: getattr(original, f.name) for f in fields(original)
                    }
                    self.layers[index] = MHCBlock.Config(
                        **values,
                        mhc_dim=self.dim,
                        mhc_streams=self.mhc["streams"],
                        mhc_iterations=self.mhc["iterations"],
                        mhc_fixed_identity=self.mhc["fixed_identity"],
                        mhc_pre_scale=self.mhc.get("pre_scale", 1.0),
                        mhc_fused_merge=self.mhc_fused_merge,
                        mhc_fused_backward=(
                            self.mhc_fused_backward
                            and self.mhc_fused_merge
                            and self.mhc["streams"] == 4
                        ),
                        mhc_fused_normalization=self.mhc_fused_normalization,
                    )

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims,
        parallelism,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Preprocess Qwen inputs while preserving the GDN metadata under CP."""
        mtp_enabled = bool(
            getattr(self.config, "mtp", None) and self.config.mtp["enabled"]
        )
        if not parallel_dims.cp_enabled and not mtp_enabled:
            return super().preprocess_inputs(
                input_dict,
                parallel_dims=parallel_dims,
                parallelism=parallelism,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
                **kwargs,
            )

        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        batch: dict[str, Any] = dict(input_dict)
        if mtp_enabled:
            from torchtitan.experiments.loop_qwen35.mtp_data import align_mtp_batch

            aligned = align_mtp_batch(batch)
            # Counts belong to the step hook; validity is encoded in labels
            # and safe future token/position/bank values used by forward.
            aligned.pop("mtp_valid_tokens")
            aligned.pop("mtp_input_valid_mask")
            batch.update(aligned)
            batch.pop("vision_bank_indices", None)
        padding_mask = batch.get("padding_mask")
        positions = batch.get("positions")
        if positions is not None:
            inner = self.config.first_full_attention_backend
            if isinstance(
                inner, (FlexInnerAttention.Config, VarlenInnerAttention.Config)
            ):
                batch["attention_masks"] = self.get_attention_masks(
                    positions=positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                )

        input_sharding = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(include_cp_axis=parallel_dims.cp_enabled),
        }
        # Vision token IDs must stay full until _prepare_multimodal_embeds has
        # scattered image/video features. Labels, positions and padding masks
        # remain contiguous CP shards.
        if parallel_dims.cp_enabled:
            input_sharding["input"] = _cp_replicated_token_id()

        mrope_positions = batch.pop("mrope_positions", None)
        if mrope_positions is None:
            rope_positions = positions
        else:
            rope_positions = mrope_positions
            input_sharding["positions"] = SpmdType(
                {
                    MeshAxisName.DP: spmd.V,
                    MeshAxisName.CP: spmd.V,
                    MeshAxisName.TP: spmd.R,
                },
                partition_spec=spmd.PartitionSpec(
                    (MeshAxisName.DP, MeshAxisName.CP), None
                ),
            )
        assert rope_positions is not None, (
            "Qwen3.5 needs RoPE positions: the batch must provide "
            "'positions' or 'mrope_positions'."
        )
        batch["positions"] = rope_positions
        if mtp_enabled:
            input_sharding.update(
                {
                    "mtp_input": input_sharding["input"],
                    "mtp_labels": input_sharding["labels"],
                    "mtp_positions": input_sharding["positions"],
                    "mtp_vision_bank_indices": input_sharding["input"],
                }
            )

        # prepare_context_parallel_input currently accepts only BlockMask
        # values, while Qwen3.5's mask dictionary also carries the GDN
        # VarlenMetadata. Shard the quadratic masks and tensors, then restore
        # the untouched global GDN metadata.
        original_masks = batch.get("attention_masks")
        block_masks: BlockMask | dict[str, BlockMask] | None
        if isinstance(original_masks, BlockMask):
            block_masks = original_masks
        elif isinstance(original_masks, dict):
            block_masks = {
                key: value
                for key, value in original_masks.items()
                if isinstance(value, BlockMask)
            }
            if not block_masks:
                block_masks = None
        else:
            block_masks = None

        cp_batch = dict(batch)
        ulysses = isinstance(
            self.config.first_full_attention_backend,
            (UlyssesCPVarlenInnerAttention.Config, UlyssesCPFlexInnerAttention.Config),
        )
        if block_masks is None or ulysses:
            cp_batch.pop("attention_masks", None)
        else:
            cp_batch["attention_masks"] = block_masks
        if parallel_dims.cp_enabled:
            cp_batch = prepare_context_parallel_input(
                cp_batch,
                input_sharding,
                parallel_dims.get_mesh("cp"),
                # Contiguous chunks preserve the order required by GDN.
                None,
                None,
            )
        if isinstance(original_masks, dict):
            sharded_masks = dict(original_masks)
            sharded_masks.update(cp_batch.get("attention_masks") or {})
            if isinstance(
                self.config.first_full_attention_backend,
                KVAllGatherCPVarlenInnerAttention.Config,
            ):
                mesh = parallel_dims.get_mesh("cp")
                sharded_masks["quadratic_attention"] = make_cp_varlen_metadata(
                    original_masks["quadratic_attention"],
                    mesh.get_local_rank(),
                    mesh.size(),
                    positions.shape[0],
                )
            cp_batch["attention_masks"] = sharded_masks
        elif isinstance(original_masks, BlockMask):
            if ulysses:
                cp_batch["attention_masks"] = original_masks

        batch = annotate_input_spmd_types(parallel_dims, cp_batch, input_sharding)
        attention_masks = batch.get("attention_masks")
        if attention_masks is not None:
            with set_current_spmd_mesh(parallel_dims.spmd_dense_mesh()):
                from torchtitan.models.qwen3_5.sharding import (
                    annotate_deltanet_cu_seqlens,
                )

                annotate_deltanet_cu_seqlens(attention_masks)

        inputs = batch.pop("input")
        labels = batch.pop("labels")
        if mtp_enabled:
            labels = (labels, batch.pop("mtp_labels"))
            # The future coordinates do not redefine document identity.
            batch["mtp_attention_masks"] = batch.get("attention_masks")
        return inputs, labels, batch

    def __init__(self, config: Config):
        super().__init__(config)
        # The Loop adapter supplies GDN all-to-all and CP-safe attention/masks.
        # The upstream parallelizer still rejects CP for unadapted models.
        self._qwen35_cp_supported = True
        self.loop_settings = config.loop
        self.loop_settings.validate(len(self.layers))
        gate_config = config.loop_gate or _loop_gate_config(
            self.config.dim, self.loop_settings
        )
        self.loop_gate = gate_config.build()
        self.loop_gate.requires_grad_(self.loop_settings.use_gate)
        self.input_shard = (
            config.input_shard.build() if config.input_shard is not None else None
        )
        self.loop_train_count = self.loop_settings.counts[0]
        self.mtp_settings = config.mtp
        if config.mtp and config.mtp["enabled"]:
            if len(config.mtp_layers) != 1:
                raise ValueError(
                    "Enabled Loop MTP requires exactly one configured MTP block"
                )
            self.mtp_layers = ModuleList(layer.build() for layer in config.mtp_layers)
            self.mtp_input_shard = (
                config.mtp_input_shard.build()
                if config.mtp_input_shard is not None
                else None
            )
        else:
            self.mtp_layers = None
        self.configure_mtp_trainability()

    def mtp_only(self):
        settings = getattr(self, "mtp_settings", None)
        return bool(
            settings and settings["enabled"] and settings["training_mode"] == "mtp_only"
        )

    def configure_mtp_trainability(self):
        if self.mtp_only():
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("mtp_layers."))
            self.train(self.training)
            self.audit_mtp_trainability()

    def audit_mtp_trainability(self):
        if self.mtp_only():
            for name, parameter in self.named_parameters():
                expected = name.startswith("mtp_layers.")
                if parameter.requires_grad != expected:
                    raise ValueError(
                        f"MTP trainability mismatch for {name}: expected requires_grad={expected}"
                    )
                if not expected and parameter.grad is not None:
                    raise ValueError(
                        f"Frozen parameter unexpectedly has a gradient: {name}"
                    )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.mtp_only():
            # Keep the root in training mode for the configured Loop schedule.
            for name, module in self.named_children():
                if name != "mtp_layers":
                    module.eval()
        return self

    def set_loop_train_count(self, count: int) -> None:
        """Set the synchronized repeat count used by the next train step."""
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        self.loop_train_count = count

    def select_loop_train_count(self, step: int) -> int:
        """Select and store a deterministic repeat count for ``step``."""
        count = self.loop_settings.count_for_step(step)
        self.loop_train_count = count
        return count

    def set_training_step(self, step: int) -> None:
        self.select_loop_train_count(step)

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
        mtp_input: torch.Tensor | None = None,
        mtp_vision_bank_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Call vision once on every rank and mask out text-only dummy features.

        Images and videos share one packed vision call. A text-only microbatch
        uses one minimal image, with no corresponding text placeholder. The
        fusion mask preserves its text embeddings and keeps the zero-gradient
        vision path in autograd, including when vision parameters are trainable.
        """
        if self.vision_encoder is None:
            inputs_embeds = super()._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                grid_thw=grid_thw,
                grid_thw_videos=grid_thw_videos,
                special_tokens=special_tokens,
            )
            if mtp_input is not None:
                return inputs_embeds, self.tok_embeddings(mtp_input)
            return inputs_embeds

        inputs_embeds = self.tok_embeddings(tokens)
        bank_indices = torch.full_like(tokens, -1)
        pixels, grids, offset = [], [], 0
        for values, grid, token_key in (
            (pixel_values, grid_thw, "image_id"),
            (pixel_values_videos, grid_thw_videos, "video_id"),
        ):
            if values is None and grid is None:
                continue
            if values is None or grid is None:
                raise ValueError(
                    f"Both pixel values and grid_thw are required for {token_key}"
                )
            if values.shape[0] == 0 and grid.shape[0] == 0:
                continue
            if special_tokens is None:
                raise ValueError("special_tokens is required for vision inputs")
            num_tokens = grid.prod(-1) // self.vision_encoder.spatial_merge_unit
            # Preserve the native checks for item counts and placeholder lengths.
            positions = get_vision_positions(
                tokens, num_tokens, special_tokens[token_key]
            )
            indices = build_vision_bank_indices(
                tokens, placeholder_id=special_tokens[token_key]
            )
            bank_indices = torch.where(indices >= 0, indices + offset, bank_indices)
            offset += sum(count for _, _, count in positions)
            pixels.append(values)
            grids.append(grid)

        if pixels:
            packed_pixels, packed_grid = torch.cat(pixels), torch.cat(grids)
        else:
            # One merged token, with valid attention inside its own tiny image.
            # Mask at fusion, not by making every vision attention entry invalid.
            merge = self.vision_encoder.spatial_merge_size
            patch_dim = self.vision_encoder.config.patch_embed_proj.in_features
            packed_pixels = torch.zeros(
                (merge * merge, patch_dim),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            packed_grid = torch.tensor([[1, merge, merge]], device=inputs_embeds.device)
            if spmd.is_type_checking():
                packed_pixels = spmd.mutate_type(
                    packed_pixels, "tp", src=spmd.R, dst=spmd.I
                )
                packed_grid = spmd.mutate_type(
                    packed_grid, "tp", src=spmd.R, dst=spmd.I
                )

        timer = getattr(self, "step_timing", None)
        with timer.phase("vision") if timer is not None else nullcontext():
            vision_embeds, _ = self._get_vision_embeds(
                packed_pixels, grid_thw=packed_grid
            )
        # This uses where(bank_indices >= 0, vision, text). All-false masks
        # still backpropagate explicit zeros through vision; do not detach or
        # skip fusion on text-only ranks, or their FSDP backward would diverge.
        inputs_embeds = gather_vision_embeds(
            inputs_embeds,
            vision_bank_VD=vision_embeds,
            vision_bank_indices_T=bank_indices,
        )
        if mtp_input is None:
            return inputs_embeds
        future_embeds = self.tok_embeddings(mtp_input)
        if mtp_vision_bank_indices is None:
            if pixels:
                raise ValueError(
                    "MTP vision inputs require globally aligned future bank indices"
                )
            mtp_vision_bank_indices = torch.full_like(mtp_input, -1)
        future_embeds = gather_vision_embeds(
            future_embeds,
            vision_bank_VD=vision_embeds,
            vision_bank_indices_T=mtp_vision_bank_indices,
        )
        return inputs_embeds, future_embeds

    def _run_layers(
        self,
        hidden: torch.Tensor,
        layers: Any,
        start: int,
        end: int,
        attention_masks: Any,
        positions: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        timer = getattr(self, "step_timing", None)
        with timer.phase("decoder") if timer is not None else nullcontext():
            for name, layer in layers.items():
                layer_idx = int(name)
                if not start <= layer_idx < end:
                    continue
                mhc = self.config.mhc
                if mhc and layer_idx == mhc["start"]:
                    from torchtitan.experiments.loop_qwen35.mhc.model import (
                        expand_streams,
                    )

                    hidden = expand_streams(hidden, mhc["streams"])
                hidden = layer(
                    hidden,
                    attention_masks,
                    positions,
                    padding_mask=padding_mask,
                )
                if mhc and layer_idx == mhc["end"] - 1:
                    from torchtitan.experiments.loop_qwen35.mhc.model import (
                        collapse_streams,
                    )

                    hidden = collapse_streams(hidden, mhc["streams"])
        return hidden

    def _forward_backbone(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_masks: dict[str, Any] | None = None,
        positions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        loop_count: int | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        with spmd_local_context("dp"):
            if inputs_embeds is not None:
                hidden = inputs_embeds
            elif self.tok_embeddings is not None:
                timer = getattr(self, "step_timing", None)
                with timer.phase("multimodal") if timer is not None else nullcontext():
                    hidden = self._prepare_multimodal_embeds(
                        tokens,
                        pixel_values=pixel_values,
                        pixel_values_videos=pixel_values_videos,
                        grid_thw=grid_thw,
                        grid_thw_videos=grid_thw_videos,
                        special_tokens=special_tokens,
                    )
            else:
                hidden = tokens

        input_shard = getattr(self, "input_shard", None)
        if input_shard is not None:
            hidden = input_shard(hidden)

        if spmd.is_type_checking() and input_shard is None:
            spmd.assert_type(
                hidden,
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                spmd.PartitionSpec(("dp", "cp"), None),
            )

        layers = self.layers
        settings = self.loop_settings
        count = self.loop_train_count if self.training else settings.eval_count
        if loop_count is not None:
            if type(loop_count) is not int or loop_count < 1:
                raise ValueError("loop_count must be a positive integer")
            count = loop_count

        gate = getattr(self, "loop_gate", None)

        hidden = self._run_layers(
            hidden,
            layers,
            0,
            settings.start,
            attention_masks,
            positions,
            padding_mask,
        )
        hidden = self._run_layers(
            hidden,
            layers,
            settings.start,
            settings.end,
            attention_masks,
            positions,
            padding_mask,
        )

        for _ in range(1, count):
            candidate = self._run_layers(
                hidden,
                layers,
                settings.start,
                settings.end,
                attention_masks,
                positions,
                padding_mask,
            )
            hidden = (
                gate(hidden, candidate)
                if gate is not None and settings.use_gate
                else candidate
            )

        # Keep the gate in the autograd graph when a K=1 training step is
        # sampled.  This makes its zero gradient explicit to DDP/FSDP and
        # matches the original Loop implementation.
        if gate is not None and settings.use_gate:
            hidden = gate(hidden)

        hidden = self._run_layers(
            hidden,
            layers,
            settings.end,
            len(self.config.layers),
            attention_masks,
            positions,
            padding_mask,
        )
        hidden = self.norm(hidden) if self.norm is not None else hidden
        return hidden

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_masks: dict[str, Any] | None = None,
        positions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        loop_count: int | None = None,
        mtp_input: torch.Tensor | None = None,
        mtp_positions: torch.Tensor | None = None,
        mtp_attention_masks: dict[str, Any] | None = None,
        mtp_vision_bank_indices: torch.Tensor | None = None,
    ):
        mtp_layers = getattr(self, "mtp_layers", None)
        if mtp_layers is not None and (mtp_input is None or mtp_positions is None):
            raise ValueError(
                "Loop MTP requires globally aligned future tokens and positions"
            )
        with torch.no_grad() if self.mtp_only() else nullcontext():
            inputs_embeds = None
            if mtp_layers is not None:
                with spmd_local_context("dp"):
                    inputs_embeds, future_embedding = self._prepare_multimodal_embeds(
                        tokens,
                        pixel_values=pixel_values,
                        pixel_values_videos=pixel_values_videos,
                        grid_thw=grid_thw,
                        grid_thw_videos=grid_thw_videos,
                        special_tokens=special_tokens,
                        mtp_input=mtp_input,
                        mtp_vision_bank_indices=mtp_vision_bank_indices,
                    )
            hidden = self._forward_backbone(
                tokens,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                grid_thw=grid_thw,
                grid_thw_videos=grid_thw_videos,
                attention_masks=attention_masks,
                positions=positions,
                padding_mask=padding_mask,
                special_tokens=special_tokens,
                inputs_embeds=inputs_embeds,
                loop_count=loop_count,
            )
        if mtp_layers is None:
            if self._skip_lm_head:
                return hidden
            return self.lm_head(hidden) if self.lm_head is not None else hidden
        if self.mtp_input_shard is not None:
            future_embedding = self.mtp_input_shard(future_embedding)
        mtp_hidden = mtp_layers[0](
            hidden,
            future_embedding,
            attention_masks if mtp_attention_masks is None else mtp_attention_masks,
            mtp_positions,
            padding_mask=padding_mask,
        )
        if self._skip_lm_head:
            return hidden, mtp_hidden
        return self.lm_head(hidden), self.lm_head(mtp_hidden)


class Qwen35LoopStateDictAdapter(Qwen35StateDictAdapter):
    """Add the Loop gate to the official Qwen3.5 HF adapter."""

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        mhc_weights = {k: v for k, v in state_dict.items() if ".mhc_" in k}
        hf_state_dict = super().to_hf(
            {k: v for k, v in state_dict.items() if k not in mhc_weights}
        )
        for key, value in mhc_weights.items():
            hf_state_dict[f"{self.hf_language_model_prefix}.{key}"] = value
        if getattr(self.model_config, "mtp", None) and self.model_config.mtp["enabled"]:
            from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import (
                MTP_HF_TO_NATIVE,
            )

            for source, target in MTP_HF_TO_NATIVE.items():
                if target not in state_dict:
                    raise ValueError(f"Missing MTP tensor during HF export: {target}")
                hf_state_dict[source] = state_dict[target]
        if "loop_gate.logits" in state_dict:
            hf_state_dict[f"{self.hf_language_model_prefix}.loop_gate.logits"] = (
                state_dict["loop_gate.logits"]
            )
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        mtp = getattr(self.model_config, "mtp", None)
        if mtp and mtp["enabled"]:
            from torchtitan.experiments.loop_qwen35.models.qwen35_mtp import (
                MTP_HF_TO_NATIVE,
                validate_mtp_weights,
            )

            self.mtp_tensor_manifest = validate_mtp_weights(
                hf_state_dict, self.model_config.mtp_layers[0]
            )
        mhc_weights = {k: v for k, v in hf_state_dict.items() if ".mhc_" in k}
        tt_state_dict = super().from_hf(
            {k: v for k, v in hf_state_dict.items() if k not in mhc_weights}
        )
        for key, value in mhc_weights.items():
            prefix = self.hf_language_model_prefix + "."
            if not key.startswith(prefix):
                raise ValueError(f"Unexpected mHC HF key: {key}")
            tt_state_dict[key[len(prefix) :]] = value
        if mtp and mtp["enabled"]:
            tt_state_dict.update(
                {
                    target: hf_state_dict[source]
                    for source, target in MTP_HF_TO_NATIVE.items()
                }
            )
        gate_key = f"{self.hf_language_model_prefix}.loop_gate.logits"
        if gate_key in hf_state_dict:
            tt_state_dict["loop_gate.logits"] = hf_state_dict[gate_key]
        return tt_state_dict


def build_loop_fsdp_mesh(dims):
    """Shard local-TP parameters over DP/CP, optionally replicating across nodes."""
    from torch.distributed.device_mesh import DeviceMesh

    shard_mesh = dims.get_optional_mesh(
        ["dp_shard", "cp"], include_singleton_axes=True
    )._flatten("loop_fsdp")
    return (
        DeviceMesh._concatenate([dims.get_mesh("dp_replicate"), shard_mesh])
        if dims.dp_replicate_enabled
        else shard_mesh
    )


def parallelize_qwen35_loop(model, *, skip_dp=False, **kwargs):
    """Apply Titan TP/CP, then FSDP2 to each rank's local TP parameters.

    Titan's SPMD modules store ordinary local tensors, whereas PyTorch's
    ``dp_mesh_dims`` API expects DTensor parameters on the full storage mesh.
    Sharding the local parameters on the DP/CP submesh preserves the existing
    explicit TP collectives and exposes plain tensors during forward.
    """
    from torchtitan.config import TORCH_DTYPE_MAP
    from torchtitan.distributed.fsdp import (
        apply_fsdp_to_decoder,
        apply_fsdp_to_vision_encoder,
    )

    dims = kwargs["parallel_dims"]
    if model.config.trainable_loop_only:
        model.requires_grad_(False)
        for name, layer in model.layers.items():
            if model.loop_settings.start <= int(name) < model.loop_settings.end:
                layer.requires_grad_(True)
        if model.loop_settings.use_gate:
            model.loop_gate.requires_grad_(True)
    model.audit_mtp_trainability()
    if model.config.freeze_vision and model.vision_encoder is not None:
        model.vision_encoder.requires_grad_(False)
    if dims.ep_enabled:
        raise NotImplementedError("The dense Qwen3.5 Loop adapter does not support EP")
    extra_layers = (
        ("mtp_layers",) if model.mtp_settings and model.mtp_settings["enabled"] else ()
    )
    model = parallelize_qwen3_5(
        model, skip_dp=True, extra_layer_names=extra_layers, **kwargs
    )
    model.audit_mtp_trainability()
    if skip_dp:
        return model
    dp_mesh = build_loop_fsdp_mesh(dims)
    training, parallelism = kwargs["training"], kwargs["parallelism"]
    fsdp_kwargs = dict(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        pp_enabled=dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
    )
    # Masked dummy inputs make vision a single unconditional call on every
    # rank, so it can stay independently sharded even for mixed text/images.
    if model.vision_encoder is not None:
        apply_fsdp_to_vision_encoder(model.vision_encoder, dp_mesh, **fsdp_kwargs)
    gate = getattr(model, "loop_gate", None)
    if gate is not None and model.loop_settings.use_gate:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        # Match LoopedTextModel._keep_in_fp32_modules_strict. A root BF16
        # policy would round logits before sigmoid even though forward casts
        # them back to FP32. Give the gate its own FP32 FSDP unit instead.
        fully_shard(
            gate,
            mesh=dp_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                cast_forward_inputs=False,
            ),
        )
    if model.config.mhc:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        from torchtitan.experiments.loop_qwen35.mhc.model import MHCRouter

        legacy_maca_fsdp = bool(
            getattr(torch.version, "maca", None)
        ) and torch.__version__.startswith("2.8.")
        for module in list(model.modules()):
            if isinstance(module, MHCRouter):
                router_kwargs = (
                    {"reshard_after_forward": False} if legacy_maca_fsdp else {}
                )
                fully_shard(
                    module,
                    mesh=dp_mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.float32,
                        reduce_dtype=torch.float32,
                        cast_forward_inputs=False,
                    ),
                    **router_kwargs,
                )
                if legacy_maca_fsdp:
                    # Nested FSDP 2.8 skips unshard during repeated AC replay.
                    # Keep small FP32 routers until the microbatch backward ends.
                    module.set_reshard_after_backward(False)
                    module._c500_reshard_after_microbatch = True
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        **fsdp_kwargs,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
        separate_lm_head=True,
        extra_layer_names=extra_layers,
    )
    if model.config.fsdp_prefetch:
        from torchtitan.experiments.loop_qwen35.fsdp_prefetch import (
            configure_loop_fsdp_prefetch,
        )

        configure_loop_fsdp_prefetch(model)
    model.audit_mtp_trainability()
    return model


def _clone_qwen35_config(
    base_config: Qwen35Model.Config,
    loop_settings: LoopSettings,
) -> Qwen35LoopModel.Config:
    config_kwargs = {
        config_field.name: getattr(base_config, config_field.name)
        for config_field in fields(Qwen35Model.Config)
        if config_field.init
    }
    config_kwargs["loop"] = loop_settings
    config_kwargs["loop_gate"] = _loop_gate_config(base_config.dim, loop_settings)
    return Qwen35LoopModel.Config(**config_kwargs)


def model_registry(
    flavor: str = "4B",
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    flash_attention_impl: str | None = None,
    loop_settings: LoopSettings | None = None,
    moe_comm_backend: str | None = None,
    converters=None,
) -> ModelSpec:
    """Return a TorchTitan ModelSpec for Qwen3.5 with Loop execution."""
    settings = loop_settings or LoopSettings()
    base_spec = qwen35_model_registry(
        flavor,
        seq_len=seq_len,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        converters=converters,
    )
    if flash_attention_impl is not None:
        if attn_backend != "varlen" or flash_attention_impl not in (
            "FA2",
            "FA3",
            "FA4",
        ):
            raise ValueError(
                "flash_attention_impl requires varlen and must be FA2, FA3 or FA4"
            )
        for layer in base_spec.model.layers:
            if layer.attention is not None:
                layer.attention.inner_attention.flash_attention_impl = (
                    flash_attention_impl
                )
    loop_config = _clone_qwen35_config(base_spec.model, settings)
    loop_config.loop.validate(len(loop_config.layers))
    return replace(
        base_spec,
        name="qwen3_5_loop",
        flavor=f"{flavor}_loop",
        model=loop_config,
        parallelize_fn=parallelize_qwen35_loop,
        pipelining_fn=None,
        state_dict_adapter=Qwen35LoopStateDictAdapter,
    )


__all__ = [
    "LoopGate",
    "LoopSettings",
    "Qwen35LoopModel",
    "Qwen35LoopStateDictAdapter",
    "model_registry",
    "parallelize_qwen35_loop",
]
