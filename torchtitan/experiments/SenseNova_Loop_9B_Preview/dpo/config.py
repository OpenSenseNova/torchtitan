# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""DPO-only options and native loss configuration."""

import math
from dataclasses import dataclass

from torchtitan.components.loss import BaseLoss

LOSS_SCALE_CHOICES = (
    "default",
    "last_round",
    "default+ignore_empty_think",
    "last_round+ignore_empty_think",
)


class DPOLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        beta: float = 0.1
        num_chunks: int = 0

    def __init__(self, config, *, compile_config=None):
        self.config = config

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "DPOLoss requires DPOTrainingEngine and complete preference pairs"
        )


def add_arguments(parser):
    parser.add_argument("--rlhf-type", choices=("dpo",))
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss-type", default="sigmoid")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--ref-model")
    parser.add_argument("--ref-checkpoint")
    parser.add_argument("--initial-rank-weights")
    parser.add_argument("--dpo-token-budget", type=int)
    parser.add_argument(
        "--loss-scale",
        "--loss_scale",
        choices=LOSS_SCALE_CHOICES,
        help="DPO assistant-token mask; defaults to default (all assistant turns)",
    )


def validate_arguments(args, parser):
    if args.rlhf_type != "dpo":
        if (
            args.ref_model
            or args.ref_checkpoint
            or args.initial_rank_weights
            or args.dpo_token_budget is not None
            or args.loss_scale is not None
            or args.beta != 0.1
            or args.loss_type != "sigmoid"
            or args.label_smoothing != 0
        ):
            parser.error(
                "Reference, rank initialization, pair token budget and loss_scale require --rlhf-type dpo"
            )
        return
    if args.loss_scale is None:
        args.loss_scale = "default"
    if not math.isfinite(args.beta) or args.beta <= 0:
        parser.error("DPO beta must be finite and positive")
    if args.loss_type != "sigmoid" or args.label_smoothing != 0:
        parser.error("Loop DPO supports only sigmoid loss with label_smoothing=0")
    if args.initial_checkpoint and args.initial_rank_weights:
        parser.error("Choose HF --initial-checkpoint or native --initial-rank-weights")
    if (
        args.packing_strategy != "greedy"
        or args.packing_length_cache
        or args.packing_prefetch_packs != "0"
    ):
        parser.error(
            "DPO supports greedy pair packing without SFT length cache or prefetch"
        )
    if args.cp < 1 or args.max_length < 1 or args.micro_batch_size < 1:
        parser.error("DPO CP, max length and micro batch size must be positive")
    if args.dpo_token_budget is None:
        args.dpo_token_budget = 2 * args.max_length * args.micro_batch_size
    if args.dpo_token_budget < 1 or args.dpo_token_budget % (
        args.cp * max(1, args.loss_chunks)
    ):
        parser.error(
            "DPO token budget must be positive and divisible by CP * max(1, LOSS_CHUNKS)"
        )


def configure_dpo(config, args, resolved):
    import hashlib
    import json
    from dataclasses import replace
    from pathlib import Path

    from torchtitan.experiments.SenseNova_Loop_9B_Preview.model_config import (
        identify_base,
    )
    from torchtitan.observability.metrics import MetricsProcessor
    from .checkpoint import DPOCheckpoint
    from .data import PreferenceDataLoader

    for source in (args.ref_model, args.ref_checkpoint):
        if not source:
            continue
        path = Path(source)
        if not path.is_dir():
            raise ValueError(f"Reference source does not exist: {source}")
        if (path / "config.json").exists():
            raw = json.loads((path / "config.json").read_text())
            reference_base = identify_base(raw, seq_len=args.max_length)[1]
            if getattr(args, "num_layers", 0):
                if reference_base["text_config.num_hidden_layers"] < args.num_layers:
                    raise ValueError("Reference has fewer layers than the policy")
                reference_base["retained_num_hidden_layers"] = args.num_layers
            if reference_base != resolved["base_model"]:
                raise ValueError(
                    "Reference base architecture/vocabulary differs from policy"
                )
            if raw.get("loop_settings") is not None:
                from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
                    _load_loop_module,
                )

                if (
                    _load_loop_module().LoopSettings(**raw["loop_settings"]).to_dict()
                    != resolved["loop_settings"]
                ):
                    raise ValueError("Reference Loop configuration differs from policy")
        elif not (path / "complete.json").exists():
            raise ValueError(
                "Explicit HF reference needs config.json; native reference needs complete.json"
            )
        if source == args.ref_model:
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "processor_config.json",
                "chat_template.jinja",
                "vocab.json",
                "merges.txt",
            ):
                policy, reference = Path(args.model) / name, path / name
                if policy.exists() != reference.exists() or (
                    policy.exists()
                    and hashlib.sha256(policy.read_bytes()).digest()
                    != hashlib.sha256(reference.read_bytes()).digest()
                ):
                    raise ValueError(
                        f"Reference tokenizer/processor/template mismatch: {name}"
                    )
    resolved["dpo"] = dict(
        algorithm="sigmoid_dpo",
        beta=args.beta,
        label_smoothing=0,
        loss_scale=args.loss_scale,
        ref_model=str(Path(args.ref_model).resolve()) if args.ref_model else "",
        ref_checkpoint=str(Path(args.ref_checkpoint).resolve())
        if args.ref_checkpoint
        else "",
        freeze_vision=args.freeze_vision,
        trainable_loop_only=args.trainable_loop_only,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        token_budget=args.dpo_token_budget,
        packing=args.packing,
        loss_chunks=args.loss_chunks,
    )
    config.training = replace(
        config.training,
        num_tokens_per_microbatch_per_dp_rank=args.dpo_token_budget,
        num_tokens_per_train_step=args.dpo_token_budget
        * args.global_batch_size
        // args.micro_batch_size,
    )
    config.loss = DPOLoss.Config(beta=args.beta, num_chunks=args.loss_chunks)
    config.dataloader = PreferenceDataLoader.Config(
        paths=args.cached_dataset,
        model_path=args.model,
        loss_scale=args.loss_scale,
        packing=args.packing,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        seed=args.seed,
    )
    config.checkpointer = DPOCheckpoint.Config(
        initial_load_path=args.initial_checkpoint or args.model,
        initial_rank_weights=args.initial_rank_weights or "",
        resolved_config=resolved,
        resume_from=args.resume_checkpoint or "",
        interval=args.save_steps,
        resume=args.resume,
        load_only=not args.save,
    )
    config.metrics = MetricsProcessor.Config(
        log_freq=args.logging_steps,
        enable_wandb=args.report_to == "wandb",
        disable_color_printing=True,
    )
