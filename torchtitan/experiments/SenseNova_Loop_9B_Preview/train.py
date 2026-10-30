# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone TorchTitan Loop SFT and DPO on conversations."""

import json
import os
import sys
from pathlib import Path

from torchtitan.experiments.SenseNova_Loop_9B_Preview.runtime import (
    configure_worker_runtime,
)

configure_worker_runtime()

import torch
import torch.distributed as dist

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.SenseNova_Loop_9B_Preview.activation_checkpoint import (
    checkpoint_config,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.checkpoint import (
    LoopCheckpoint,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.config import parse_args
from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
    qwen35_loop,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.data import ArrowDataLoader
from torchtitan.experiments.SenseNova_Loop_9B_Preview.metrics import (
    LoopMetricsProcessor,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.model_config import (
    resolve_model_config,
    validate_parallelism,
)
from torchtitan.observability.logging import init_logger


def main(argv=None):
    args = parse_args(argv)
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.model_config import (
        resolve_hf_path,
    )

    args.model = resolve_hf_path(args.model)
    if args.initial_checkpoint:
        args.initial_checkpoint = resolve_hf_path(args.initial_checkpoint)
    if args.ref_model:
        args.ref_model = resolve_hf_path(args.ref_model)

    init_logger()
    native_metadata = None
    if args.mtp_num_layers:
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.checkpoint import (
            select_native_checkpoint,
        )

        native_path = select_native_checkpoint(
            Path(args.output_dir) / "checkpoints",
            resume=args.resume,
            resume_from=args.resume_checkpoint or "",
        )
        if native_path is not None:
            native_metadata = json.loads((native_path / "complete.json").read_text())
    resolved = resolve_model_config(
        args.model,
        seq_len=args.max_length,
        loop_config=args.loop_config,
        initial_checkpoint=args.initial_checkpoint,
        mtp_num_layers=args.mtp_num_layers,
        mtp_training_mode=args.mtp_training_mode,
        mtp_loss_scaling_factor=args.mtp_loss_scaling_factor,
        freeze_vision=args.freeze_vision,
        native_resume_metadata=native_metadata,
    )
    if args.num_layers:
        if (
            not 1
            <= args.num_layers
            <= resolved["base_model"]["text_config.num_hidden_layers"]
        ):
            raise ValueError(
                "num-layers must be positive and no larger than the HF decoder"
            )
        resolved["base_model"]["retained_num_hidden_layers"] = args.num_layers
    validate_parallelism(
        resolved,
        tp=args.tp,
        cp=args.cp,
        tokens_per_microbatch=args.max_length * args.micro_batch_size,
        loss_chunks=args.loss_chunks,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Runtime: python={sys.executable} torch={torch.__version__} "
            f"attention={args.attention_backend} flash_attention_impl={args.flash_attention_impl} packing={args.packing} "
            f"packing_strategy={args.packing_strategy} packing_buffer_size={args.packing_buffer_size} "
            f"padding_free={args.padding_free} tp={args.tp} cp={args.cp} "
            f"cp_gdn_overlap={args.cp_gdn_overlap} "
            f"cp_attention={args.cp_attention_backend if args.cp > 1 else 'disabled'} "
            f"cp_load_balance={'heads' if args.cp > 1 and args.cp_attention_backend == 'ulysses' else 'none'} "
            f"model_flavor={resolved['model_flavor']} dp_replicate={args.dp_replicate} dp_shard={args.dp_shard} "
            f"sequence_parallel={args.sequence_parallel} deterministic={args.deterministic} "
            f"activation_reduce={args.activation_reduce_dtype} "
            f"activation_checkpoint_policy={args.activation_checkpoint_policy} "
            f"gdn_output={args.gdn_output_kernel}",
            flush=True,
        )
        print(
            f"Resolved Loop: {json.dumps(resolved['loop_settings'])}; "
            f"sources={resolved['loop_sources']}",
            flush=True,
        )
    if args.mtp_num_layers and int(os.environ.get("RANK", "0")) == 0:
        print(f"Resolved MTP: {json.dumps(resolved['mtp'])}", flush=True)
    if args.flash_attention_impl == "FA4":
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.attention_backend import (
            install_fa4_causal_window_compat,
        )

        install_fa4_causal_window_compat()
    config = qwen35_loop(
        resolved,
        seq_len=args.max_length,
        attn_backend="varlen" if args.attention_backend == "flash" else "flex",
        flash_attention_impl=args.flash_attention_impl
        if args.attention_backend == "flash"
        else None,
    )
    if args.num_layers:
        config.model_spec.model.layers = config.model_spec.model.layers[
            : args.num_layers
        ]
        config.model_spec.model.loop.validate(args.num_layers)
    mhc = None
    if args.mhc:
        loop = resolved["loop_settings"]
        if not loop["start"] <= args.mhc_start < args.mhc_end <= loop["end"]:
            raise ValueError(
                "mHC interval must lie completely inside the repeated Loop interval"
            )
        mhc = dict(
            start=args.mhc_start,
            end=args.mhc_end,
            streams=args.mhc_streams,
            iterations=args.mhc_iterations,
            fixed_identity=args.mhc_fixed_identity,
            pre_scale=args.mhc_pre_scale,
        )
    resolved["experiment"] = dict(
        mhc=mhc,
        lr=args.lr,
        router_lr=args.router_lr if args.mhc else None,
        router_warmup=args.router_warmup_steps if args.mhc else None,
        warmup_ratio=args.warmup_ratio,
        decay_ratio=args.decay_ratio,
        min_lr_factor=args.min_lr_factor,
        train_iters=args.train_iters,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_length=args.max_length,
        global_batch_size=args.global_batch_size,
    )
    resolved["execution"] = dict(
        flash_attention_impl=args.flash_attention_impl
        if args.attention_backend == "flash"
        else None,
        cp_attention_backend=args.cp_attention_backend,
        cp_gdn_overlap=args.cp_gdn_overlap,
        cp_load_balance="heads"
        if args.cp > 1 and args.cp_attention_backend == "ulysses"
        else "none",
        mhc_fused_merge=args.mhc_fused_merge,
        mhc_fused_backward=args.mhc_fused_backward,
        mhc_fused_normalization=args.mhc_fused_normalization,
        packing_prefetch_packs=args.packing_prefetch_packs,
        packing_prefetch_mode=args.packing_prefetch_mode,
        packing_prefetch_max_mib=args.packing_prefetch_max_mib,
        packing_length_cache=args.packing_length_cache,
        packing_dp_balance=args.packing_dp_balance,
        packing_vision_balance=args.packing_vision_balance,
        packing_vision_weight=args.packing_vision_weight,
        fsdp_prefetch=args.fsdp_prefetch,
        activation_checkpoint=args.activation_checkpoint,
        activation_checkpoint_policy=args.activation_checkpoint_policy,
        diagnostics_interval=args.diagnostics_interval,
        step_timing=args.step_timing,
        profile_start_step=args.profile_start_step,
        profile_steps=args.profile_steps,
        profile_ranks=args.profile_ranks,
        profile_with_stack=args.profile_with_stack,
        profile_memory=args.profile_memory,
    )
    config.model_spec.model.mhc = mhc
    config.model_spec.model.mhc_fused_merge = args.mhc_fused_merge
    config.model_spec.model.mhc_fused_backward = args.mhc_fused_backward
    config.model_spec.model.mhc_fused_normalization = args.mhc_fused_normalization
    config.model_spec.model.fsdp_prefetch = args.fsdp_prefetch
    config.model_spec.model.cp_attention_backend = args.cp_attention_backend
    config.model_spec.model.cp_gdn_overlap = args.cp_gdn_overlap
    config.comm.init_timeout_seconds = args.init_timeout_seconds
    config.comm.train_timeout_seconds = args.train_timeout_seconds
    config.hf_assets_path = args.model
    config.model_spec.model.freeze_vision = args.freeze_vision
    config.model_spec.model.trainable_loop_only = args.trainable_loop_only
    config.dump_folder = args.output_dir
    parallel_kwargs = dict(
        tensor_parallel_degree=args.tp,
        context_parallel_degree=args.cp,
        pipeline_parallel_degree=args.pp,
        expert_parallel_degree=args.ep,
        enable_sequence_parallel=args.sequence_parallel,
        data_parallel_replicate_degree=args.dp_replicate,
        data_parallel_shard_degree=args.dp_shard,
        context_parallel_load_balancer=None,
    )
    parallel_kwargs["activation_reduce_dtype"] = (
        None
        if args.activation_reduce_dtype == "input"
        else args.activation_reduce_dtype
    )
    config.parallelism = ParallelismConfig(**parallel_kwargs)
    config.training = TrainingConfig(
        max_context_length=args.max_length,
        num_tokens_per_microbatch_per_dp_rank=args.max_length * args.micro_batch_size,
        num_tokens_per_train_step=args.max_length * args.global_batch_size,
        steps=args.train_iters,
        dtype="float32",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        max_norm=args.clip_grad,
        disable_cuda_graphs=True,
    )
    config.debug = DebugConfig(seed=args.seed, deterministic=args.deterministic)
    config.activation_checkpoint = checkpoint_config(
        args.activation_checkpoint, args.activation_checkpoint_policy
    )
    cross_entropy = CrossEntropyLoss.Config(
        global_vocab_size=config.model_spec.model.vocab_size
    )
    if resolved["mtp"]["enabled"]:
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.loss import (
            LoopMTPLoss,
        )

        cross_entropy = LoopMTPLoss.Config(
            global_vocab_size=config.model_spec.model.vocab_size,
            training_mode=resolved["mtp"]["training_mode"],
            loss_scaling_factor=resolved["mtp"]["loss_scaling_factor"],
        )
    config.loss = (
        ChunkedLossWrapper.Config(num_chunks=args.loss_chunks, loss_fn=cross_entropy)
        if args.loss_chunks
        else cross_entropy
    )
    config.optimizer = default_adamw(lr=args.lr, weight_decay=args.weight_decay)
    from torchtitan.components.optimizer import ParamGroupConfig
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.mhc.schedule import (
        MHCSchedulers,
    )

    train_routers = args.mhc and not (
        resolved["mtp"]["enabled"] and args.mtp_training_mode == "mtp_only"
    )
    if train_routers:
        config.optimizer.param_groups.insert(
            0,
            ParamGroupConfig(
                pattern=r"\.mhc_(attn|ffn)\.",
                optimizer_name="AdamW",
                optimizer_kwargs=dict(
                    lr=args.router_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
                ),
            ),
        )
    schedule_steps = (
        native_metadata["mtp"]["effective_schedule_steps"]
        if native_metadata is not None and resolved["mtp"]["enabled"]
        else args.train_iters
    )
    config.lr_scheduler = MHCSchedulers.Config(
        warmup_steps=round(schedule_steps * args.warmup_ratio),
        router_warmup_steps=args.router_warmup_steps if train_routers else 0,
        min_lr_factor=args.min_lr_factor,
        total_steps=schedule_steps,
        decay_ratio=args.decay_ratio,
    )
    config.metrics = LoopMetricsProcessor.Config(
        log_freq=args.logging_steps,
        enable_wandb=args.report_to == "wandb",
        disable_color_printing=True,
    )
    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_exp_name:
            os.environ["WANDB_RUN_NAME"] = args.wandb_exp_name
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.packing_vision import (
        resolve_vision_weights,
    )

    vision_weights = (
        resolve_vision_weights(
            resolved,
            cp=args.cp,
            freeze_vision=args.freeze_vision,
            activation_checkpoint=args.activation_checkpoint,
            checkpoint_policy=args.activation_checkpoint_policy,
            scale=args.packing_vision_weight,
        )
        if args.packing_vision_balance
        else {}
    )
    resolved["execution"]["packing_vision_weights"] = (
        vision_weights if args.packing_vision_balance else None
    )
    config.dataloader = ArrowDataLoader.Config(
        paths=args.cached_dataset,
        model_path=args.model,
        packing=args.packing,
        padding_free=args.padding_free,
        seed=args.seed,
        packing_strategy=args.packing_strategy,
        packing_dp_balance=args.packing_dp_balance,
        packing_buffer_size=args.packing_buffer_size,
        packing_length_cache=args.packing_length_cache,
        packing_prefetch_packs=args.packing_prefetch_packs,
        packing_prefetch_mode=args.packing_prefetch_mode,
        packing_prefetch_max_mib=args.packing_prefetch_max_mib,
        packing_vision_balance=args.packing_vision_balance,
        packing_vision_weights=vision_weights,
        grad_accumulation=args.grad_accumulation,
    )
    config.checkpointer = LoopCheckpoint.Config(
        initial_load_path=args.initial_checkpoint or args.model,
        resume_from=args.resume_checkpoint or "",
        resolved_config=resolved,
        interval=args.save_steps,
        resume=args.resume,
        load_only=not args.save,
    )
    if args.rlhf_type == "dpo":
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.dpo.config import (
            configure_dpo,
        )

        configure_dpo(config, args, resolved)
    if args.inspect_data:
        tokenizer = config.tokenizer.build(tokenizer_path=args.model)
        loader = config.dataloader.build(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            max_context_length=args.max_length,
            num_tokens_per_microbatch=config.training.num_tokens_per_microbatch_per_dp_rank,
        )
        batch = loader.next_step()[0][0] if args.rlhf_type == "dpo" else next(loader)
        print(
            json.dumps(
                {
                    "tokens": batch.input.numel(),
                    "supervised_tokens": batch.num_valid_tokens,
                    "documents": int(
                        ((batch.positions == 0) & ~batch.padding_mask).sum()
                    ),
                    "images": len(batch.model_kwargs.get("grid_thw", [])),
                    "padding": int(batch.padding_mask.sum()),
                    "torch": torch.__version__,
                    "python": sys.executable,
                }
            ),
            flush=True,
        )
        return
    if args.gdn_output_kernel == "stable":
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.kernels import (
            stabilize_gdn_output,
        )

        stabilize_gdn_output()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        (Path(args.output_dir) / "launcher_args.json").write_text(
            json.dumps(vars(args), indent=2)
        )
    trainer = None
    try:
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.trainer import (
            SFTTrainer,
        )

        if args.rlhf_type == "dpo":
            from torchtitan.experiments.SenseNova_Loop_9B_Preview.dpo.trainer import (
                DPOTrainer,
            )

            trainer = DPOTrainer(config, dpo_args=args)
        else:
            trainer = SFTTrainer(config, experiment_args=args)
        trainer.train()
    finally:
        if trainer is not None:
            trainer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
