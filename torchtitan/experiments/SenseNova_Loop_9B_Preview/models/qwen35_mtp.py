# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Pretrained Qwen3.5 MTP block, separate from the repeated backbone layers."""

import copy
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

import spmd_types as spmd
import torch

from torchtitan.models.common import Linear
from torchtitan.models.qwen3_5.model import OffsetRMSNorm, Qwen35TransformerBlock

# T = packed tokens, D = hidden width.


class Qwen35MTPBlock(Qwen35TransformerBlock):
    """Normalize future embedding and final backbone hidden, then predict once."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35TransformerBlock.Config):
        enorm: OffsetRMSNorm.Config
        hnorm: OffsetRMSNorm.Config
        eh_proj: Linear.Config
        mtp_norm: OffsetRMSNorm.Config

    def __init__(self, config):
        if (
            config.attention is None
            or config.delta_net is not None
            or config.moe is not None
        ):
            raise ValueError("Qwen MTP requires one dense full-attention block")
        super().__init__(config)
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.eh_proj = config.eh_proj.build()
        self.mtp_norm = config.mtp_norm.build()

    def forward(
        self, hidden_TD, embedding_TD, attention_masks, positions, *, padding_mask=None
    ):
        fused_TD = self.eh_proj(
            torch.cat((self.enorm(embedding_TD), self.hnorm(hidden_TD)), dim=-1)
        )
        output_TD = super().forward(
            fused_TD, attention_masks, positions, padding_mask=padding_mask
        )
        return self.mtp_norm(output_TD)


def build_mtp_config(base_config):
    """Deep-copy a full-attention block without mutating the backbone config."""
    full = next(
        (layer for layer in base_config.layers if layer.attention is not None), None
    )
    if full is None or full.moe is not None:
        raise ValueError(
            "Pretrained Qwen MTP requires a dense full-attention configuration"
        )
    values = {
        field.name: copy.deepcopy(getattr(full, field.name))
        for field in fields(Qwen35TransformerBlock.Config)
        if field.init
    }
    # Sharding is configured explicitly for the two MTP inputs during update.
    values["sharding_config"] = None
    return Qwen35MTPBlock.Config(
        **values,
        enorm=copy.deepcopy(base_config.norm),
        hnorm=copy.deepcopy(base_config.norm),
        eh_proj=Linear.Config(
            in_features=2 * base_config.dim,
            out_features=base_config.dim,
            param_init=copy.deepcopy(full.attention.wq.param_init),
        ),
        mtp_norm=copy.deepcopy(base_config.norm),
    )


def set_mtp_sharding_config(config, *, enable_sp, embedding_layout):
    """Use native Qwen TP layouts and a replicated fusion projection."""
    from torchtitan.models.common.decoder_sharding import (
        dense_activation_placement,
        dense_param_placement,
        dense_sequence_parallel_placement,
        norm_config,
        pre_lm_head_norm_config,
        token_id_placement,
    )
    from torchtitan.models.qwen3_5.sharding import set_qwen35_layer_sharding
    from torchtitan.protocols.sharding import ShardingConfig

    full_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    sequence_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    config.sharding_config = ShardingConfig(
        in_src_shardings={
            "hidden_TD": full_layout,
            "embedding_TD": embedding_layout,
            "padding_mask": token_id_placement(),
        },
        in_dst_shardings={
            "hidden_TD": sequence_layout,
            "embedding_TD": sequence_layout,
            "padding_mask": token_id_placement(enable_sp=enable_sp),
        },
        out_src_shardings=full_layout,
    )
    set_qwen35_layer_sharding(
        config,
        attention_input_layout=sequence_layout,
        enable_sp=enable_sp,
        enable_ep=False,
    )
    # Off-SP residual gradients are already invariant. Head-parallel QK norms
    # retain native TP R because their head contributions must still be summed.
    for name in ("enorm", "hnorm", "attention_norm"):
        norm = norm_config(enable_sp=enable_sp)
        norm.in_src_shardings = {"x": sequence_layout}
        getattr(config, name).sharding_config = norm
    config.mtp_norm.sharding_config = pre_lm_head_norm_config(enable_sp=enable_sp)
    config.mtp_norm.sharding_config.in_src_shardings = {"x": sequence_layout}
    config.eh_proj.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": sequence_layout},
        out_src_shardings=sequence_layout,
    )


MTP_HF_TO_NATIVE = {
    "mtp.pre_fc_norm_embedding.weight": "mtp_layers.0.enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "mtp_layers.0.hnorm.weight",
    "mtp.fc.weight": "mtp_layers.0.eh_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight": "mtp_layers.0.attn.wq.weight",
    "mtp.layers.0.self_attn.k_proj.weight": "mtp_layers.0.attn.wk.weight",
    "mtp.layers.0.self_attn.v_proj.weight": "mtp_layers.0.attn.wv.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "mtp_layers.0.attn.wo.weight",
    "mtp.layers.0.self_attn.q_norm.weight": "mtp_layers.0.attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight": "mtp_layers.0.attn.k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight": "mtp_layers.0.feed_forward.w1.weight",
    "mtp.layers.0.mlp.up_proj.weight": "mtp_layers.0.feed_forward.w3.weight",
    "mtp.layers.0.mlp.down_proj.weight": "mtp_layers.0.feed_forward.w2.weight",
    "mtp.layers.0.input_layernorm.weight": "mtp_layers.0.attention_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight": "mtp_layers.0.ffn_norm.weight",
    "mtp.norm.weight": "mtp_layers.0.mtp_norm.weight",
}


def tensor_sha256(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def parameter_manifest(model):
    """Describe the actual frozen/trainable sets without reading tensor values."""
    from torchtitan.components.checkpointer.utils import canonical_fqn

    parameters = {
        canonical_fqn(name): value for name, value in model.named_parameters()
    }
    trainable = sorted(
        name for name, value in parameters.items() if value.requires_grad
    )
    frozen = sorted(parameters.keys() - set(trainable))
    result = {}
    for category, names in (("trainable", trainable), ("frozen", frozen)):
        result[f"{category}_parameters"] = names
        result[f"{category}_parameters_sha256"] = hashlib.sha256(
            "\n".join(names).encode()
        ).hexdigest()
        result[f"{category}_numel"] = sum(parameters[name].numel() for name in names)
    return result


def validate_mtp_weights(weights, config):
    """Validate the complete source tensors before casting or partitioning."""
    actual = {key for key in weights if key.startswith("mtp.")}
    expected = set(MTP_HF_TO_NATIVE)
    if actual != expected:
        raise ValueError(
            f"Invalid pretrained MTP keys: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    with torch.device("meta"):
        shapes = {
            f"mtp_layers.0.{name}": tuple(value.shape)
            for name, value in config.build().state_dict().items()
        }
    manifest = {}
    for source, target in MTP_HF_TO_NATIVE.items():
        value = weights[source]
        if tuple(value.shape) != shapes[target]:
            raise ValueError(
                f"MTP tensor shape mismatch for {source}: {tuple(value.shape)} != {shapes[target]}"
            )
        if value.dtype != torch.bfloat16:
            raise ValueError(
                f"MTP HF initialization requires BF16 source tensors, got {value.dtype} for {source}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite pretrained MTP tensor: {source}")
        manifest[source] = {
            "target": target,
            "shape": list(value.shape),
            "dtype": "bfloat16",
            "finite": True,
            "sha256": tensor_sha256(value),
        }
    return manifest


def mtp_source_manifest(source, tensors):
    """Record the source configuration, index and exact MTP contents."""
    source = Path(source).resolve()
    files = {}
    for name in ("config.json", "model.safetensors.index.json"):
        path = source / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "weight_format": "qwen3_5_hf_bf16",
        "version": 1,
        "source_path": str(source),
        "files": files,
        "tensors": tensors,
    }


def slice_mtp_for_tp(model, name, value, tp_mesh):
    """Use the declared TP placement, never infer a partition from shapes."""
    from torchtitan.protocols.sharding import resolve_placements

    module_name, _, parameter_name = name.rpartition(".")
    # FeedForward exposes logical w1/w3 state keys for its interleaved w13
    # parameter. Each logical projection has the same declared TP axis.
    if module_name.endswith((".feed_forward.w1", ".feed_forward.w3")):
        module_name = module_name.rsplit(".", 1)[0] + ".w13"
    module = model.get_submodule(module_name)
    sharding = getattr(module, "_sharding_config", None)
    if sharding is None or parameter_name not in sharding.state_shardings:
        raise ValueError(f"Missing declared MTP placement for {name}")
    placements = resolve_placements(sharding.state_shardings[parameter_name], tp_mesh)
    if tp_mesh.mesh_dim_names != ("tp",) or len(placements) != 1:
        raise ValueError("MTP TP slicing requires the named one-dimensional tp mesh")
    (placement,) = placements
    if placement.is_shard():
        return torch.tensor_split(value, tp_mesh.size(), dim=placement.dim)[
            tp_mesh.get_local_rank()
        ].contiguous()
    if not placement.is_replicate():
        raise ValueError(f"Unsupported MTP parameter placement for {name}: {placement}")
    return value
