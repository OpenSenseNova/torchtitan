# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan config entry points for Loop and mHC."""

from __future__ import annotations

import copy


def _load_loop_module():
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.models import qwen35_loop

    return qwen35_loop


def qwen35_loop(
    resolved,
    *,
    seq_len: int,
    attn_backend: str = "flex",
    flash_attention_impl: str | None = None,
):
    """Build a registered HF base with experiment-specific Loop settings."""
    from torchtitan.components.tokenizer import MultiModalTokenizer
    from torchtitan.config import TrainingConfig
    from torchtitan.models.qwen3_5 import QWEN3_5_SPECIAL_TOKENS
    from torchtitan.trainer import Trainer

    loop_module = _load_loop_module()
    spec = loop_module.model_registry(
        resolved["model_flavor"],
        seq_len=seq_len,
        loop_settings=loop_module.LoopSettings(**resolved["loop_settings"]),
        attn_backend=attn_backend,
        flash_attention_impl=flash_attention_impl,
    )
    mtp = resolved.get("mtp", {})
    if mtp.get("enabled"):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.models.qwen35_mtp import (
            build_mtp_config,
        )

        spec.model.mtp = copy.deepcopy(mtp)
        spec.model.mtp_layers = [build_mtp_config(spec.model)]
        if mtp["training_mode"] == "mtp_only":
            spec.model.freeze_vision = True
    config = Trainer.Config(
        model_spec=spec,
        hf_assets_path=resolved["model_path"],
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        training=TrainingConfig(max_context_length=seq_len, disable_cuda_graphs=True),
    )
    if mtp.get("enabled"):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.loss import (
            LoopMTPLoss,
        )

        config.loss = LoopMTPLoss.Config(
            global_vocab_size=spec.model.vocab_size,
            training_mode=mtp["training_mode"],
            loss_scaling_factor=mtp["loss_scaling_factor"],
        )
    return config


__all__ = ["qwen35_loop"]
