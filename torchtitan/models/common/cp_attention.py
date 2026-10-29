# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernels.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TYPE_CHECKING

import spmd_types as spmd

import torch

from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group, spmd_redistribute

from torchtitan.models.common.attention import (
    FlexInnerAttention,
    VarlenInnerAttention,
    VarlenMetadata,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

__all__ = [
    "CPInnerAttention",
    "KVAllGatherCPFlexInnerAttention",
    "KVAllGatherCPVarlenInnerAttention",
    "UlyssesCPInnerAttention",
    "UlyssesCPFlexInnerAttention",
    "UlyssesCPVarlenInnerAttention",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1


class CPInnerAttention(ABC):
    """Inner attention that owns its context-parallel behavior."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @classmethod
    @abstractmethod
    def cp_shard(
        cls,
        input_dict: dict[str, Any],
        input_shardings: dict[str, Any] | None,
        cp_mesh: "DeviceMesh",
        load_balancer_type: str | None,
        ptrr_mask_key: str | None,
    ) -> dict[str, Any]:
        """Shard model inputs for this attention implementation."""


class KVAllGatherCPFlexInnerAttention(CPInnerAttention, FlexInnerAttention):
    """FlexInnerAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, FlexInnerAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    @classmethod
    def cp_shard(
        cls,
        input_dict: dict[str, Any],
        input_shardings: dict[str, Any] | None,
        cp_mesh: "DeviceMesh",
        load_balancer_type: str | None,
        ptrr_mask_key: str | None,
    ) -> dict[str, Any]:
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        return prepare_context_parallel_input(
            input_dict,
            input_shardings,
            cp_mesh,
            load_balancer_type,
            ptrr_mask_key,
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP attention requires an active multi-rank CP mesh axis."
            )
        k_THK, v_THV = (
            spmd_redistribute(
                x,
                cp_group,
                src=spmd.S(_TOKEN_DIM),
                dst=spmd.R,
                backward_options={"op_dtype": self.reduce_dtype},
            )
            for x in (k_THK, v_THV)
        )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)


class CPVarlenMetadata(NamedTuple):
    local: VarlenMetadata
    kv_start: int
    kv_end: int


def make_cp_varlen_metadata(
    metadata: VarlenMetadata, rank: int, size: int, num_tokens: int
) -> CPVarlenMetadata:
    """Align each local Q segment with its causal KV prefix.

    FA2 aligns unequal Q/K lengths at the right edge. Keeping K only through
    the last local query therefore preserves global causal positions, including
    documents split across CP ranks. Contiguous CP needs only one KV slice.
    Host metadata construction happens once per microbatch, outside the kernel.
    """
    if size < 1 or num_tokens < size or num_tokens % size or not 0 <= rank < size:
        raise ValueError("CP requires equal, nonempty contiguous token shards")
    boundaries = metadata.cu_seq_q.tolist()
    if (
        len(boundaries) < 2
        or boundaries != metadata.cu_seq_k.tolist()
        or boundaries[0] != 0
        or boundaries[-1] != num_tokens
        or any(b < a for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError(
            "CP varlen expects self-attention metadata covering the complete token batch"
        )
    start, end = rank * (num_tokens // size), (rank + 1) * (num_tokens // size)
    segments = [
        (a, b)
        for a, b in zip(boundaries, boundaries[1:])
        if b > start and a < end and b > a
    ]
    if not segments:
        raise ValueError("CP rank has no attention segments")
    q_lengths = [min(b, end) - max(a, start) for a, b in segments]
    k_lengths = [min(b, end) - a for a, b in segments]
    q_offsets, k_offsets = [0], [0]
    for q_length, k_length in zip(q_lengths, k_lengths):
        q_offsets.append(q_offsets[-1] + q_length)
        k_offsets.append(k_offsets[-1] + k_length)
    return CPVarlenMetadata(
        VarlenMetadata(
            metadata.cu_seq_q.new_tensor(q_offsets),
            metadata.cu_seq_k.new_tensor(k_offsets),
            max(q_lengths),
            max(k_lengths),
        ),
        segments[0][0],
        end,
    )


class KVAllGatherCPVarlenInnerAttention(CPInnerAttention, VarlenInnerAttention):
    """Causal variable-length FlashAttention with local Q and gathered K/V.

    Unlike head-sharded CP, this also supports a single local KV head under TP.
    KV gradients are reduce-scattered in FP32. This is an all-gather baseline,
    not a ring-attention implementation; KV storage still spans the sequence.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, VarlenInnerAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if self.window_size[1] != 0:
            raise ValueError(
                "KV-all-gather varlen CP currently supports causal attention only"
            )
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    @classmethod
    def cp_shard(
        cls, input_dict, input_shardings, cp_mesh, load_balancer_type, ptrr_mask_key
    ):
        if load_balancer_type is not None:
            raise ValueError(
                "Varlen KV-all-gather CP requires contiguous shards without load balancing"
            )
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        masks = input_dict.pop("attention_masks")
        num_tokens = input_dict["positions"].shape[0]
        prepare_context_parallel_input(input_dict, input_shardings, cp_mesh, None, None)
        convert = lambda mask: make_cp_varlen_metadata(
            mask, cp_mesh.get_local_rank(), cp_mesh.size(), num_tokens
        )
        input_dict["attention_masks"] = (
            {key: convert(mask) for key, mask in masks.items()}
            if isinstance(masks, dict)
            else convert(masks)
        )
        return input_dict

    def forward(
        self, q_THK, k_THK, v_THV, *, attention_masks: CPVarlenMetadata, **kwargs
    ):
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None or not isinstance(attention_masks, CPVarlenMetadata):
            raise ValueError(
                "CP varlen attention requires an active CP mesh and local CP metadata"
            )
        k_THK, v_THV = (
            spmd_redistribute(
                x,
                cp_group,
                src=spmd.S(_TOKEN_DIM),
                dst=spmd.R,
                backward_options={"op_dtype": self.reduce_dtype},
            )
            for x in (k_THK, v_THV)
        )
        span = slice(attention_masks.kv_start, attention_masks.kv_end)
        return super().forward(
            q_THK,
            k_THK[span].contiguous(),
            v_THV[span].contiguous(),
            attention_masks=attention_masks.local,
            **kwargs,
        )


class UlyssesCPInnerAttention(CPInnerAttention):
    """Move CP sharding between the token and head dimensions."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    @classmethod
    def cp_shard(
        cls,
        input_dict: dict[str, Any],
        input_shardings: dict[str, Any] | None,
        cp_mesh: "DeviceMesh",
        load_balancer_type: str | None,
        ptrr_mask_key: str | None,
    ) -> dict[str, Any]:
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        attention_masks = input_dict.pop("attention_masks", None)
        prepare_context_parallel_input(
            input_dict,
            input_shardings,
            cp_mesh,
            None,
            None,
        )
        if attention_masks is not None:
            input_dict["attention_masks"] = attention_masks
        return input_dict

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP attention requires an active multi-rank CP mesh axis."
            )
        # Shard heads instead of tokens: (T/cp, H, *) -> (T, H/cp, *).
        q_THK, k_THK, v_THV = (
            spmd.redistribute(
                x,
                cp_group,
                src=spmd.S(_TOKEN_DIM),
                dst=spmd.S(_HEAD_DIM),
            )
            for x in (q_THK, k_THK, v_THV)
        )
        # super() follows the concrete class MRO to its inner attention.
        # pyrefly: ignore [missing-attribute]
        out_THV = super().forward(q_THK, k_THK, v_THV, **kwargs)
        # Back to sharded tokens: (T, H/cp, V) -> (T/cp, H, V).
        return spmd.redistribute(
            out_THV,
            cp_group,
            src=spmd.S(_HEAD_DIM),
            dst=spmd.S(_TOKEN_DIM),
        )


class UlyssesCPFlexInnerAttention(UlyssesCPInnerAttention, FlexInnerAttention):
    """FlexInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPInnerAttention.Config, FlexInnerAttention.Config):
        pass


class UlyssesCPVarlenInnerAttention(UlyssesCPInnerAttention, VarlenInnerAttention):
    """VarlenInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPInnerAttention.Config, VarlenInnerAttention.Config):
        pass
