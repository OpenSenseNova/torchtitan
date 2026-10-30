# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve a supported HF base and independent Loop settings without loading weights."""

import copy
import json
import math
from pathlib import Path

from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
    _load_loop_module,
)
from torchtitan.models.qwen3_5 import qwen3_5_configs

# These are the dense architectures implemented by the local Titan adapter.
# Loop intervals and counts are deliberately not part of this registry.
DENSE_FLAVORS = ("0.8B", "2B", "4B", "9B", "27B")


def resolve_mtp_config(
    *,
    num_layers=0,
    training_mode="mtp_only",
    loss_scaling_factor=None,
    freeze_vision=True,
):
    """Resolve explicit MTP options without enabling them from HF metadata."""
    if type(num_layers) is not int or num_layers not in (0, 1):
        raise ValueError("MTP supports exactly 0 or 1 physical layers")
    if training_mode not in ("mtp_only", "joint"):
        raise ValueError("MTP training mode must be mtp_only or joint")
    factor = loss_scaling_factor
    if factor is not None and (not math.isfinite(factor) or factor < 0):
        raise ValueError("MTP loss scaling factor must be finite and nonnegative")
    if num_layers == 0:
        return {
            "enabled": False,
            "num_layers": 0,
            "training_mode": "disabled",
            "loss_scaling_factor": 0.0,
        }
    if factor is None:
        factor = 1.0 if training_mode == "mtp_only" else 0.1
    if training_mode == "mtp_only":
        if factor == 0:
            raise ValueError("mtp_only requires a positive MTP loss scaling factor")
        if not freeze_vision:
            raise ValueError(
                "mtp_only freezes all non-MTP parameters; freeze_vision=false conflicts with this mode"
            )
    return {
        "enabled": True,
        "num_layers": 1,
        "training_mode": training_mode,
        "loss_scaling_factor": float(factor),
        "prediction_offset": 1,
        "teacher_forcing": True,
        "normalization": "per_objective_global_valid_tokens",
        "hidden_stage": "post_final_norm",
        "detach_backbone": training_mode == "mtp_only",
        "automatic_unfreezing": False,
    }


def validate_mtp_source_config(config, mtp, *, model_flavor):
    """Validate the supported pretrained architecture before tensor import."""
    if not mtp["enabled"]:
        return
    if model_flavor != "9B":
        raise ValueError(
            "Pretrained Loop MTP currently supports the dense Qwen3.5-9B architecture"
        )
    text = config.get("text_config", config)
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("MTP initialization requires unquantized HF safetensors")
    if text.get("mtp_num_hidden_layers") != 1:
        raise ValueError("Pretrained source must declare exactly one MTP layer")
    if text.get("mtp_use_dedicated_embeddings", False):
        raise ValueError(
            "Dedicated MTP embeddings are unsupported; MTP must share the existing embedding/head"
        )


def read_json_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _native_architecture(model):
    attention = next(
        layer.attention for layer in model.layers if layer.attention is not None
    )
    delta = next(
        layer.delta_net for layer in model.layers if layer.delta_net is not None
    )
    vision = model.vision_encoder
    return {
        "model_type": "qwen3_5",
        "text_config.model_type": "qwen3_5_text",
        "text_config.hidden_size": model.dim,
        "text_config.num_hidden_layers": len(model.layers),
        "text_config.vocab_size": model.vocab_size,
        "text_config.intermediate_size": model.layers[0].feed_forward.w2.in_features,
        "text_config.num_attention_heads": attention.n_heads,
        "text_config.num_key_value_heads": attention.n_kv_heads,
        "text_config.head_dim": attention.head_dim,
        "text_config.rms_norm_eps": model.norm.eps,
        "text_config.attention_bias": attention.wq.bias,
        "text_config.attention_dropout": 0.0,
        "text_config.attn_output_gate": True,
        "text_config.hidden_act": "silu",
        "text_config.layer_types": [
            "full_attention" if layer.attention is not None else "linear_attention"
            for layer in model.layers
        ],
        "text_config.linear_num_key_heads": delta.in_proj_q.out_features
        // delta.key_head_dim,
        "text_config.linear_num_value_heads": delta.in_proj_v.out_features
        // delta.value_head_dim,
        "text_config.linear_key_head_dim": delta.key_head_dim,
        "text_config.linear_value_head_dim": delta.value_head_dim,
        "text_config.linear_conv_kernel_dim": delta.conv_kernel_size,
        "text_config.rope_parameters.rope_type": "default",
        "text_config.rope_parameters.rope_theta": attention.rope.theta,
        "text_config.rope_parameters.partial_rotary_factor": attention.rotary_dim
        / attention.head_dim,
        "text_config.rope_parameters.mrope_section": attention.rope.mrope_section,
        "text_config.rope_parameters.mrope_interleaved": True,
        "vision_config.depth": vision.num_layers,
        "vision_config.hidden_size": vision.dim,
        "vision_config.intermediate_size": vision.block.mlp.fc1.out_features,
        "vision_config.num_heads": vision.num_heads,
        "vision_config.out_hidden_size": vision.merger.fc2.out_features,
        "vision_config.in_channels": vision.in_channels,
        "vision_config.patch_size": vision.patch_size,
        "vision_config.temporal_patch_size": vision.temporal_patch_size,
        "vision_config.spatial_merge_size": vision.spatial_merge_size,
        "vision_config.num_position_embeddings": vision.num_position_embeddings,
        "vision_config.hidden_act": "gelu_pytorch_tanh",
        "vision_config.deepstack_visual_indexes": [],
    }


def identify_base(config, *, seq_len):
    if config.get("model_type") != "qwen3_5":
        raise ValueError(
            "This Loop entry supports registered dense Qwen3.5 bases (model_type=qwen3_5)"
        )
    if config.get("quantization_config"):
        raise ValueError(
            "Quantized HF weights are not supported by this full-parameter Loop entry"
        )
    text = dict(config["text_config"])
    # Normalize optional HF fields; model dimensions always come from config.json.
    for key, value in {
        "model_type": "qwen3_5_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 4,
    }.items():
        text.setdefault(key, value)
    if "layer_types" not in text:
        interval = text.get("full_attention_interval", 4)
        if type(interval) is not int or interval < 1:
            raise ValueError("full_attention_interval must be a positive integer")
        text["layer_types"] = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(text["num_hidden_layers"])
        ]
    rope = dict(text.get("rope_parameters") or text.get("rope_scaling") or {})
    rope.setdefault("rope_type", rope.pop("type", "default"))
    rope.setdefault("rope_theta", text.get("rope_theta", 10_000_000.0))
    rope.setdefault("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))
    rope.setdefault("mrope_interleaved", True)
    text["rope_parameters"] = rope
    vision = dict(config["vision_config"])
    vision.setdefault("hidden_act", "gelu_pytorch_tanh")
    vision.setdefault("deepstack_visual_indexes", [])
    normalized = dict(config, text_config=text, vision_config=vision)

    for flavor in DENSE_FLAVORS:
        factory, limit = qwen3_5_configs[flavor]
        model = factory(attn_backend="flex", seq_len=1)
        if (model.dim, len(model.layers)) != (
            text["hidden_size"],
            text["num_hidden_layers"],
        ):
            continue
        expected = _native_architecture(model)
        actual = {}
        for key in expected:
            value = normalized
            for component in key.split("."):
                if not isinstance(value, dict) or component not in value:
                    raise ValueError(
                        f"HF config is missing required architecture field {key}"
                    )
                value = value[component]
            actual[key] = value
        mismatches = [key for key in expected if actual[key] != expected[key]]
        if mismatches:
            raise ValueError(
                f"HF config does not match the supported {flavor} architecture: {', '.join(mismatches)}"
            )
        if seq_len > min(limit, text.get("max_position_embeddings", limit)):
            raise ValueError(
                f"Requested sequence length {seq_len} exceeds this base model's context limit"
            )
        # Titan trains an independent head, but the HF initialization contract
        # must still agree when importing/resuming weights.
        actual["tie_word_embeddings"] = config.get(
            "tie_word_embeddings", text.get("tie_word_embeddings", False)
        )
        return flavor, actual
    raise ValueError(
        f"No supported dense Qwen3.5 base matches hidden_size={text['hidden_size']}, "
        f"num_hidden_layers={text['num_hidden_layers']}; registered sizes: {', '.join(DENSE_FLAVORS)}"
    )


def resolve_model_config(
    model_path,
    *,
    seq_len,
    loop_config=None,
    initial_checkpoint=None,
    mtp_num_layers=0,
    mtp_training_mode="mtp_only",
    mtp_loss_scaling_factor=None,
    freeze_vision=True,
    native_resume_metadata=None,
):
    model_file = Path(model_path) / "config.json"
    base = read_json_object(model_file)
    flavor, architecture = identify_base(base, seq_len=seq_len)
    values, sources = {}, []

    def merge(settings, source):
        if not isinstance(settings, dict):
            raise ValueError(f"Loop settings must be a JSON object in {source}")
        values.update(settings)
        sources.append(str(source))

    if "loop_settings" in base:
        merge(base["loop_settings"], model_file)
    if initial_checkpoint and native_resume_metadata is None:
        checkpoint_file = Path(initial_checkpoint) / "config.json"
        if (
            checkpoint_file.resolve() != model_file.resolve()
            and checkpoint_file.is_file()
        ):
            checkpoint = read_json_object(checkpoint_file)
            _, checkpoint_architecture = identify_base(checkpoint, seq_len=seq_len)
            if checkpoint_architecture != architecture:
                raise ValueError(
                    "INITIAL_CHECKPOINT and MODEL_PATH have different base architectures"
                )
            if "loop_settings" in checkpoint:
                merge(checkpoint["loop_settings"], checkpoint_file)
    if native_resume_metadata is not None:
        merge(native_resume_metadata["topology"]["loop"], "native checkpoint")
    if loop_config:
        merge(read_json_object(loop_config), loop_config)
    if not sources:
        raise ValueError(
            "No Loop configuration: pass --loop-config or provide config.json.loop_settings"
        )
    settings = _load_loop_module().LoopSettings(**values)
    settings.validate(architecture["text_config.num_hidden_layers"])
    mtp = resolve_mtp_config(
        num_layers=mtp_num_layers,
        training_mode=mtp_training_mode,
        loss_scaling_factor=mtp_loss_scaling_factor,
        freeze_vision=freeze_vision,
    )
    saved_mtp = None
    if native_resume_metadata is not None:
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.checkpoint import (
            normalized_mtp_metadata,
        )

        saved_mtp = normalized_mtp_metadata(native_resume_metadata)
        if saved_mtp["enabled"] != mtp["enabled"]:
            raise ValueError("Native resume cannot enable or disable MTP")
    else:
        validate_mtp_source_config(base, mtp, model_flavor=flavor)
    if mtp["enabled"]:
        if flavor != "9B":
            raise ValueError(
                "Pretrained Loop MTP requires the dense Qwen3.5-9B architecture"
            )
        if saved_mtp is None:
            source = Path(initial_checkpoint or model_path).resolve()
            validate_mtp_source_config(
                read_json_object(source / "config.json"), mtp, model_flavor=flavor
            )
        else:
            source = saved_mtp["source_manifest"]["source_path"]
            mtp["source_manifest"] = copy.deepcopy(saved_mtp["source_manifest"])
        mtp.update(
            source_path=str(source),
            weight_format="qwen3_5_hf_bf16",
            weight_format_version=1,
            main_loss_role="monitor"
            if mtp_training_mode == "mtp_only"
            else "objective",
        )
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_flavor": flavor,
        "base_model": architecture,
        "loop_settings": settings.to_dict(),
        "loop_sources": sources,
        "mtp": mtp,
    }


def validate_parallelism(
    resolved, *, tp, cp, tokens_per_microbatch=None, loss_chunks=0
):
    if type(tp) is not int or type(cp) is not int or min(tp, cp) < 1:
        raise ValueError("TP and CP must be positive integers")
    base = resolved["base_model"]
    for field in ("num_attention_heads", "num_key_value_heads"):
        heads = base[f"text_config.{field}"]
        if heads % tp:
            raise ValueError(f"TP={tp} must divide {field}={heads}")
    for field in ("linear_num_key_heads", "linear_num_value_heads"):
        heads = base[f"text_config.{field}"]
        if heads % (tp * cp):
            raise ValueError(f"TP*CP={tp * cp} must divide {field}={heads}")
    if resolved.get("mtp", {}).get("enabled"):
        vocab = base["text_config.vocab_size"]
        if vocab % tp:
            raise ValueError(
                f"MTP shared head requires TP={tp} to divide vocabulary={vocab}"
            )
        if type(loss_chunks) is not int or loss_chunks < 0:
            raise ValueError("MTP loss chunks must be a nonnegative integer")
        if tokens_per_microbatch is not None:
            if type(tokens_per_microbatch) is not int or tokens_per_microbatch <= 0:
                raise ValueError("MTP token microbatch must be a positive integer")
            divisor = tp * cp * max(1, loss_chunks)
            if tokens_per_microbatch % divisor:
                raise ValueError(
                    "MTP token microbatch must be divisible by TP*CP*max(1, loss chunks)"
                )


def resolve_hf_path(source):
    """Resolve local HF assets or download a public/private Hub model snapshot."""
    path = Path(source).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if path.is_absolute() or str(source).startswith((".", "~")):
        raise FileNotFoundError(f"HF model directory does not exist: {source}")
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=source,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.txt",
            "*.model",
            "*.tiktoken",
            "*.safetensors",
        ],
    )
