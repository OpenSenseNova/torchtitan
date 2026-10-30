# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import spmd_types as spmd
import torch
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.components.loss import CrossEntropyLoss, IGNORE_INDEX
from torchtitan.config import CompileConfig, FSDPSymmMemScope, ParallelismConfig
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    current_spmd_mesh,
)
from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.linear import Linear

from torchtitan.models.common.mtp import roll_mtp_sequence
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import ModuleList

if TYPE_CHECKING:
    from torch.distributed.fsdp import DataParallelMeshDims


class MTPTransformerBlock(TransformerBlock):
    """Generic MTP block for decoder-only transformer models.

    The block implements the DeepSeek-V3 style fusion:

    ``eh_proj(cat(enorm(shifted_embedding), hnorm(previous_hidden)))``

    followed by one regular transformer block.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        eh_proj: Linear.Config
        mtp_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.eh_proj = config.eh_proj.build()
        self.mtp_norm = config.mtp_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        mtp_input_embed: torch.Tensor,
        prev_embed: torch.Tensor,
        mtp_input_valid_mask: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        mtp_padding_mask_T = ~mtp_input_valid_mask
        if padding_mask is not None:
            mtp_padding_mask_T = mtp_padding_mask_T | padding_mask
        prev_embed = prev_embed * mtp_input_valid_mask.unsqueeze(-1).to(
            dtype=prev_embed.dtype
        )
        h = self.eh_proj(
            torch.cat([self.enorm(mtp_input_embed), self.hnorm(prev_embed)], dim=-1)
        )
        h = h + self.attention(self.attention_norm(h), attention_masks, positions)
        if self.moe_enabled:
            h = h + self.moe(
                self.ffn_norm(h),
                padding_mask_T=mtp_padding_mask_T,
            )
        else:
            h = h + self.feed_forward(self.ffn_norm(h))
        return self.mtp_norm(h)


class MTPDecoder(Decoder):
    """Decoder variant that owns MTP layers.

    MTP is kept as model behavior: the main decoder consumes the normal input
    sequence, and each MTP layer predicts one extra depth from preprocessed
    shifted token embeddings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        mtp_layers: list = field(default_factory=list)

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            if len(self.mtp_layers) <= 0:
                return Decoder.Config.update_from_config(
                    self,
                    config=config,
                    **kwargs,
                )

            num_main_layers = len(self.layers)
            self.layers.extend(self.mtp_layers)
            try:
                Decoder.Config.update_from_config(self, config=config, **kwargs)
            finally:
                del self.layers[num_main_layers:]

            parallelism = config.parallelism
            # TODO: Add Pipeline Parallel support for MTP.
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "MTP does not support pipeline parallelism yet."
                )

    def __init__(self, config: Config):
        super().__init__(config)
        if not config.mtp_layers:
            self.mtp_layers = None
            return

        self.mtp_layers = ModuleList()
        for layer_config in config.mtp_layers:
            if not isinstance(layer_config, MTPTransformerBlock.Config):
                raise ValueError(
                    "MTPDecoder requires Config.mtp_layers to contain "
                    "MTPTransformerBlock.Config instances."
                )
            self.mtp_layers.append(layer_config.build())

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
        dict[str, Any],
    ]:
        """Prepare aligned pairs before applying CP sharding and annotations."""
        del kwargs
        # Function-local import avoids a circular import
        # (context_parallel.api -> models.common -> decoder).
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        batch: dict[str, Any] = dict(input_dict)
        tokens = batch["input"]
        labels = batch["labels"]
        positions = batch.get("positions")
        padding_mask = batch.get("padding_mask")
        if self.mtp_layers is not None and positions is None:
            raise ValueError("MTP input preprocessing requires positions.")

        depths = (
            range(1, len(self.mtp_layers) + 1)
            if self.mtp_layers is not None
            else range(0)
        )
        input_sharding = decoder_input_sharding()
        for depth in depths:
            mtp_input_tokens, mtp_input_valid_mask = roll_mtp_sequence(
                tokens,
                shift=depth,
                positions=positions,
                padding_mask=padding_mask,
                fill_value=0,
                return_valid_mask=True,
            )
            mtp_labels = roll_mtp_sequence(
                labels,
                shift=depth,
                positions=positions,
                padding_mask=padding_mask,
                fill_value=IGNORE_INDEX,
                return_valid_mask=False,
            )
            batch[f"mtp_input_tokens_{depth}"] = mtp_input_tokens
            batch[f"mtp_labels_{depth}"] = mtp_labels
            batch[f"mtp_input_valid_mask_{depth}"] = mtp_input_valid_mask
            input_sharding[f"mtp_input_tokens_{depth}"] = input_sharding["input"]
            input_sharding[f"mtp_labels_{depth}"] = input_sharding["labels"]
            input_sharding[f"mtp_input_valid_mask_{depth}"] = input_sharding["input"]

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

        if parallel_dims.cp_enabled:
            batch = prepare_context_parallel_input(
                batch,
                input_sharding,
                parallel_dims.get_mesh("cp"),
                parallelism.context_parallel_load_balancer,
                parallelism.context_parallel_ptrr_mask_key,
            )
        batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)

        main_tokens = batch.pop("input")
        main_labels = batch.pop("labels")
        if self.mtp_layers is None:
            return main_tokens, main_labels, batch

        input_tokens = (
            main_tokens,
            *(batch.pop(f"mtp_input_tokens_{depth}") for depth in depths),
        )
        loss_labels = (
            main_labels,
            *(batch.pop(f"mtp_labels_{depth}") for depth in depths),
        )
        batch["mtp_input_valid_masks"] = tuple(
            batch.pop(f"mtp_input_valid_mask_{depth}") for depth in depths
        )
        return input_tokens, loss_labels, batch

    def forward(
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, ...],
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        mtp_input_valid_masks: tuple[torch.Tensor, ...] | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        if self.mtp_layers is None:
            if not isinstance(tokens, torch.Tensor):
                raise ValueError("A decoder without MTP expects one token tensor.")
            return super().forward(
                tokens,
                positions,
                attention_masks,
                padding_mask=padding_mask,
            )
        if self.tok_embeddings is None:
            raise ValueError("MTP decoder forward requires token embeddings.")
        if not isinstance(tokens, tuple) or len(tokens) != len(self.mtp_layers) + 1:
            raise ValueError(
                "MTP decoder requires one main token tensor and one prepared "
                "token tensor per MTP layer."
            )
        if mtp_input_valid_masks is None or len(mtp_input_valid_masks) != len(
            self.mtp_layers
        ):
            raise ValueError("MTP decoder requires one validity mask per MTP layer.")

        main_tokens, *mtp_input_tokens = tokens

        # Keep this aligned with Decoder.forward(), but preserve the pre-norm
        # hidden state because MTP consumes the last decoder-layer output.
        h = self.tok_embeddings(main_tokens)
        for layer in self.layers.values():
            h = layer(h, attention_masks, positions, padding_mask=padding_mask)

        prev_depth_hidden = h
        h = self.norm(h) if self.norm is not None else h

        mtp_outputs = []
        for layer, depth_tokens, mtp_input_valid_mask in zip(
            self.mtp_layers,
            mtp_input_tokens,
            mtp_input_valid_masks,
            strict=True,
        ):
            mtp_input_embed = self.tok_embeddings(depth_tokens)
            prev_depth_hidden = layer(
                mtp_input_embed,
                prev_depth_hidden,
                mtp_input_valid_mask,
                attention_masks,
                positions,
                padding_mask=padding_mask,
            )
            mtp_outputs.append(prev_depth_hidden)

        outputs = (h, *mtp_outputs)
        if self._skip_lm_head:
            predictions = outputs
        else:
            predictions = tuple(
                self.lm_head(item) if self.lm_head is not None else item
                for item in outputs
            )
        return predictions


def apply_fsdp_to_mtp_decoder(
    model: MTPDecoder,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
    edp_mesh_dims: DataParallelMeshDims | None = None,
    symm_mem_scope: FSDPSymmMemScope = None,
) -> None:
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=pp_enabled,
        cpu_offload=cpu_offload,
        reshard_after_forward_policy=reshard_after_forward_policy,
        ep_degree=ep_degree,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=symm_mem_scope,
        extra_layer_names=("mtp_layers",) if model.mtp_layers is not None else (),
    )


class MTPLoss(CrossEntropyLoss):
    """DeepSeek-V3 weighted multi-term cross-entropy objective."""

    @dataclass(kw_only=True, slots=True)
    class Config(CrossEntropyLoss.Config):
        mtp_scale: float = 0.3

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        super().__init__(config, compile_config=compile_config)
        self.mtp_scale = config.mtp_scale

    def __call__(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        global_valid_tokens: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted objective from aligned prediction/label pairs."""
        del loss_inputs
        if not isinstance(pred, tuple) or not isinstance(labels, tuple):
            raise ValueError("MTPLoss expects prediction and labels tuples.")
        if len(pred) != len(labels):
            raise ValueError(
                "MTPLoss requires one labels tensor per prediction, "
                f"got {len(pred)} predictions and {len(labels)} labels."
            )
        num_mtp_layers = len(pred) - 1
        if num_mtp_layers <= 0:
            raise ValueError(
                "MTPLoss expects a main prediction and at least one auxiliary "
                "prediction."
            )
        mtp_weight = self.mtp_scale / num_mtp_layers
        main_loss, _ = super().__call__(pred[0], labels[0])
        mtp_loss = pred[0].new_zeros((), dtype=torch.float32)
        if spmd.is_type_checking():
            mtp_loss = spmd.mutate_type(
                mtp_loss,
                src=spmd.R,
                dst={"dp": spmd.P, "cp": spmd.P, "tp": spmd.I},
            )
        for mtp_pred, mtp_labels in zip(pred[1:], labels[1:], strict=True):
            depth_loss, _ = super().__call__(mtp_pred, mtp_labels)
            mtp_loss = mtp_loss + depth_loss * mtp_weight
        loss = main_loss + mtp_loss
        if global_valid_tokens is not None:
            if current_spmd_mesh() is not None:
                spmd.assert_type(
                    global_valid_tokens,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
                )
            loss = loss / global_valid_tokens
        return loss, {}
