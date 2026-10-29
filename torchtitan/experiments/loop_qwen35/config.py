# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line configuration for Loop SFT, DPO and MTP."""

import argparse
import math
import os


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Use true or false")
    return value.lower() == "true"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--loop-config",
        default=os.getenv("LOOP_CONFIG") or None,
        help="Loop JSON overrides; otherwise read config.json.loop_settings",
    )
    parser.add_argument(
        "--dataset",
        "--cached-dataset",
        dest="cached_dataset",
        nargs="+",
        required=True,
        help="Local JSONL files or Arrow dataset directories",
    )
    parser.add_argument("--initial-checkpoint")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=0,
        help="Retain the first N decoder layers for regression; 0 keeps the full model",
    )
    parser.add_argument("--dp-replicate", type=int, default=1)
    parser.add_argument("--dp-shard", type=int, default=-1)
    parser.add_argument(
        "--diagnostics-interval",
        type=int,
        default=1,
        help="Router statistics every N steps; 0 disables optional statistics",
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=0,
        help="Absolute step; 0 disables profiling",
    )
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--profile-ranks", type=int, nargs="+", default=[0])
    parser.add_argument("--profile-with-stack", type=boolean, default=True)
    parser.add_argument("--profile-memory", type=boolean, default=True)
    parser.add_argument(
        "--resume-checkpoint",
        help="External native checkpoint to resume when output has no completed checkpoint",
    )
    for name, default in (
        ("tp", 1),
        ("cp", 1),
        ("pp", 1),
        ("ep", 1),
        ("etp", 1),
        ("vpp", 1),
        ("max-length", 262144),
        ("micro-batch-size", 1),
        ("global-batch-size", 1),
        ("train-iters", 100),
        ("loss-chunks", 256),
        ("save-steps", 100),
        ("seed", 42),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (
        ("packing", None),
        ("padding-free", True),
        ("freeze-vision", True),
        ("sequence-parallel", True),
        ("deterministic", False),
        ("activation-checkpoint", True),
        ("resume", False),
        ("save", True),
    ):
        parser.add_argument("--" + name, type=boolean, default=default)
    parser.add_argument(
        "--attention-backend", choices=("flash", "flex"), default="flash"
    )
    parser.add_argument(
        "--flash-attention-impl",
        choices=("FA2", "FA4"),
        default="FA2",
        help="FlashAttention implementation; FA4 requires deterministic=false",
    )
    parser.add_argument(
        "--activation-checkpoint-policy",
        choices=("full", "attention"),
        default="full",
        help="Retain FlashAttention outputs with attention; recompute everything with full",
    )
    parser.add_argument(
        "--cp-attention-backend",
        choices=("ulysses", "all_gather"),
        default="ulysses",
        help="Ulysses balances CP attention across heads; all_gather retains the baseline",
    )
    parser.add_argument(
        "--cp-gdn-overlap",
        type=boolean,
        default=True,
        help="Overlap native GDN CP exchanges with Q/K/V convolutions",
    )
    parser.add_argument(
        "--activation-reduce-dtype", choices=("input", "float32"), default="float32"
    )
    parser.add_argument(
        "--gdn-output-kernel", choices=("stable", "native"), default="stable"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument(
        "--decay-ratio",
        type=float,
        help="Decay ratio; defaults to 1.0 for SFT and 0.0 for DPO",
    )
    parser.add_argument("--decay-type", choices=("cosine",), default="cosine")
    parser.add_argument("--min-lr-factor", type=float, default=0.0)
    parser.add_argument("--trainable-loop-only", type=boolean, default=False)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument(
        "--step-timing",
        type=boolean,
        default=False,
        help="Log per-step host/CUDA phase times and rank min/mean/max at logging boundaries",
    )
    parser.add_argument(
        "--wandb-project", default=os.getenv("WANDB_PROJECT", "qwen35-loop-titan")
    )
    parser.add_argument(
        "--wandb-exp-name", default=os.getenv("WANDB_RUN_NAME", os.getenv("WANDB_NAME"))
    )
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--mhc", type=boolean, default=False)
    parser.add_argument("--mhc-start", type=int, default=4)
    parser.add_argument("--mhc-end", type=int, default=8)
    parser.add_argument("--mhc-streams", type=int, choices=(2, 4), default=2)
    parser.add_argument("--mhc-iterations", type=int, default=20)
    parser.add_argument("--mhc-fixed-identity", type=boolean, default=False)
    parser.add_argument(
        "--mhc-fused-merge",
        type=boolean,
        default=False,
        help="Fuse BF16 four-stream residual merge; requires N4 mHC",
    )
    parser.add_argument(
        "--mhc-fused-normalization",
        type=boolean,
        default=False,
        help="Fuse router casts and pointwise work while preserving native FP32 reductions",
    )
    parser.add_argument(
        "--mhc-fused-backward",
        type=boolean,
        default=None,
        help="Fuse N4 backward; enabled by default when fused N4 merge is enabled",
    )
    parser.add_argument(
        "--fsdp-prefetch",
        type=boolean,
        default=False,
        help="Prefetch Loop decoder blocks and their FP32 mHC routers",
    )
    parser.add_argument("--router-lr", type=float, default=4e-5)
    parser.add_argument("--router-warmup-steps", type=int, default=20)
    parser.add_argument("--val-dataset", nargs="+")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument(
        "--mhc-pre-scale",
        type=float,
        default=1.0,
        help="Positive residual branch scale for mHC",
    )
    parser.add_argument(
        "--init-timeout-seconds",
        type=int,
        default=int(os.getenv("LOOP_DEBUG_INIT_TIMEOUT", "1800")),
    )
    parser.add_argument(
        "--train-timeout-seconds",
        type=int,
        default=int(os.getenv("LOOP_DEBUG_TRAIN_TIMEOUT", "1800")),
    )
    parser.add_argument("--packing-strategy", choices=("greedy", "balanced"))
    parser.add_argument(
        "--packing-dp-balance",
        type=boolean,
        default=True,
        help="For balanced packing, group whole packs by cost across DP and GA; "
        "false preserves packing/prefetch but randomly assigns complete packs",
    )
    parser.add_argument("--packing-buffer-size", type=int, default=8192)
    parser.add_argument("--packing-length-cache", default="")
    parser.add_argument(
        "--packing-vision-balance",
        type=boolean,
        default=False,
        help="Balance decoder and vision costs using an offline image-geometry cache",
    )
    parser.add_argument(
        "--packing-vision-weight",
        type=float,
        default=1.0,
        help="Multiplier for model-derived vision FLOPs; default 1",
    )
    parser.add_argument(
        "--packing-prefetch-packs",
        default="0",
        help="CPU encoding lookahead: auto follows gradient accumulation for balanced packing; "
        "a nonnegative integer sets the depth, and 0 disables prefetch",
    )
    parser.add_argument(
        "--packing-prefetch-mode",
        choices=("row", "pack"),
        default="row",
        help="Prefetch encoded rows or complete CPU packs; pack mode preserves the data plan",
    )
    parser.add_argument(
        "--packing-prefetch-max-mib",
        type=int,
        default=1024,
        help="Ready-pack byte budget; excludes one in-flight pack and consumed batches",
    )
    parser.add_argument("--mtp-num-layers", type=int, default=0)
    parser.add_argument(
        "--mtp-training-mode", choices=("mtp_only", "joint"), default="mtp_only"
    )
    parser.add_argument("--mtp-loss-scaling-factor", type=float)
    from torchtitan.experiments.loop_qwen35.dpo.config import (
        add_arguments,
        validate_arguments,
    )

    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.decay_ratio is None:
        args.decay_ratio = 0.0 if args.rlhf_type == "dpo" else 1.0
    if args.packing is None:
        args.packing = args.rlhf_type != "dpo"
    if args.packing_strategy is None:
        args.packing_strategy = "greedy"
    validate_arguments(args, parser)
    if args.flash_attention_impl == "FA4" and (
        args.attention_backend != "flash" or args.deterministic
    ):
        parser.error("FA4 requires the flash backend and deterministic=false")
    if args.mhc_fused_backward is None:
        args.mhc_fused_backward = (
            args.mhc and args.mhc_streams == 4 and args.mhc_fused_merge
        )
    if args.mhc_fused_backward and not args.mhc_fused_merge:
        parser.error("Fused mHC backward requires --mhc-fused-merge true")
    if args.mhc_fused_merge and (not args.mhc or args.mhc_streams != 4):
        parser.error("Fused mHC merge requires --mhc true --mhc-streams 4")
    if args.mhc_fused_normalization and not args.mhc:
        parser.error("Fused mHC normalization requires --mhc true")
    if not math.isfinite(args.mhc_pre_scale) or args.mhc_pre_scale <= 0:
        parser.error("MHC pre-scale must be finite and positive")
    if min(args.init_timeout_seconds, args.train_timeout_seconds) <= 0:
        parser.error("Communication timeouts must be positive")
    if args.tp != 1 or args.pp != 1:
        parser.error("The mHC training entry currently supports TP=PP=1")
    if args.activation_checkpoint_policy == "attention" and (
        not args.activation_checkpoint or args.attention_backend != "flash"
    ):
        parser.error(
            "Attention checkpoint policy requires activation checkpointing and the flash backend"
        )
    if (
        args.profile_start_step < 0
        or args.profile_steps < 1
        or min(args.profile_ranks) < 0
    ):
        parser.error("Invalid profiler step/rank selection")
    if args.diagnostics_interval < 0:
        parser.error("Diagnostic intervals must be nonnegative")
    if args.resume_checkpoint and not args.resume:
        parser.error("--resume-checkpoint requires --resume true")
    if (
        args.mhc_iterations < 1
        or args.router_lr <= 0
        or args.eval_steps < 1
        or args.eval_batches < 1
    ):
        parser.error("Router iterations/LR and evaluation settings must be positive")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.decay_ratio <= 1:
        parser.error("WARMUP_RATIO must be in [0,1) and DECAY_RATIO in [0,1]")
    if not 0 <= args.min_lr_factor <= 1:
        parser.error("MIN_LR_FACTOR must be in [0,1]")
    world = int(os.environ.get("WORLD_SIZE", str(args.tp * args.cp * args.pp)))
    if max(args.profile_ranks) >= world:
        parser.error("Profiler ranks must be smaller than WORLD_SIZE")
    if (
        args.profile_start_step
        and args.profile_start_step + args.profile_steps - 1 > args.train_iters
    ):
        parser.error("Profiler interval must fit within TRAIN_ITERS")
    if (
        min(
            args.tp,
            args.cp,
            args.pp,
            args.ep,
            args.etp,
            args.vpp,
            args.max_length,
            args.micro_batch_size,
            args.global_batch_size,
            args.train_iters,
            args.save_steps,
            args.logging_steps,
        )
        < 1
    ):
        parser.error("Degrees, token/batch counts and steps must be positive")
    if args.ep != 1 or args.etp != 1:
        parser.error("This Qwen3.5 Loop entry is dense: EP=ETP=1")
    if args.pp != 1 or args.vpp != 1:
        parser.error("This Loop training entry requires PP=VPP=1")
    if world % (args.tp * args.cp * args.pp):
        parser.error("WORLD_SIZE must be divisible by TP*CP*PP")
    if not args.padding_free:
        parser.error("Native Titan uses flat THD inputs; PADDING_FREE must be true")
    dp = world // (args.tp * args.cp * args.pp)
    if args.dp_replicate < 1 or (args.dp_shard != -1 and args.dp_shard < 1):
        parser.error("DP_REPLICATE must be positive; DP_SHARD must be positive or -1")
    if dp % args.dp_replicate or (
        args.dp_shard != -1 and args.dp_replicate * args.dp_shard != dp
    ):
        parser.error("DP_REPLICATE*DP_SHARD must equal WORLD_SIZE/(TP*CP*PP)")
    if args.global_batch_size % (args.micro_batch_size * dp):
        parser.error("GBS must be divisible by MBS*DP")
    args.grad_accumulation = args.global_batch_size // (args.micro_batch_size * dp)
    if args.packing_prefetch_packs == "auto":
        args.packing_prefetch_packs = (
            args.global_batch_size // (args.micro_batch_size * dp)
            if args.packing_strategy == "balanced"
            else 0
        )
    else:
        try:
            args.packing_prefetch_packs = int(args.packing_prefetch_packs)
        except ValueError:
            parser.error("PACKING_PREFETCH_PACKS must be auto or a nonnegative integer")
    if args.rlhf_type != "dpo" and (
        args.max_length
        * args.micro_batch_size
        % (args.tp * args.cp * max(1, args.loss_chunks))
    ):
        parser.error("Token microbatch must be divisible by TP*CP*LOSS_CHUNKS")
    if args.loss_chunks < 0:
        parser.error("LOSS_CHUNKS must be nonnegative (0 disables chunking)")
    if args.packing_vision_balance and args.packing_strategy != "balanced":
        parser.error("Visual balance requires balanced packing")
    if not math.isfinite(args.packing_vision_weight) or args.packing_vision_weight < 0:
        parser.error("packing-vision-weight must be finite and nonnegative")
    if args.packing_buffer_size < 1:
        parser.error("PACKING_BUFFER_SIZE must be positive")
    if args.packing_prefetch_max_mib < 1:
        parser.error("packing-prefetch-max-mib must be positive")
    if args.packing_prefetch_mode == "pack" and not args.packing_prefetch_packs:
        parser.error(
            "Final pack prefetch requires --packing-prefetch-packs auto or a positive integer"
        )
    if args.packing_prefetch_packs < 0 or (
        args.packing_prefetch_packs and args.packing_strategy != "balanced"
    ):
        parser.error(
            "PACKING_PREFETCH_PACKS must be nonnegative and requires balanced packing"
        )
    if args.packing_strategy == "balanced" and (
        not args.packing or args.packing_buffer_size < dp
    ):
        parser.error(
            "Balanced packing requires packing=true and PACKING_BUFFER_SIZE >= DP"
        )
    from torchtitan.experiments.loop_qwen35.model_config import resolve_mtp_config

    try:
        mtp = resolve_mtp_config(
            num_layers=args.mtp_num_layers,
            training_mode=args.mtp_training_mode,
            loss_scaling_factor=args.mtp_loss_scaling_factor,
            freeze_vision=args.freeze_vision,
        )
    except ValueError as error:
        parser.error(str(error))
    if mtp["enabled"]:
        if args.rlhf_type == "dpo":
            parser.error("MTP cannot be combined with DPO")

        if args.trainable_loop_only:
            parser.error("MTP cannot be combined with trainable-loop-only")
    return args
