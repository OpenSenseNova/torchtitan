# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Strict full/chunk/CP/DP/packing gradient comparison with real Loop layers."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT)]

import torch

from dpo_reference_kernels import install_fp32_reference_kernels
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
    _load_loop_module,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.dpo.config import DPOLoss
from torchtitan.experiments.SenseNova_Loop_9B_Preview.dpo.data import (
    PreferenceDataLoader,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.dpo.engine import (
    DPOTrainingEngine,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.dpo.metrics import (
    add_metrics,
    reduce_metrics,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.dpo.reference import (
    ReferenceRuntime,
    local_tensor,
)
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import get_spmd_context
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator


def pair(index, image=False):
    generator = torch.Generator().manual_seed(100 + index)
    rows = []
    for side, length in enumerate((57 + 10 * index, 61 + 10 * index)):
        tokens = torch.randint(0, 120, (length,), generator=generator)
        tokens[:20] = torch.arange(20) + 10
        labels = tokens.roll(-1)
        labels[:19] = labels[-1] = -100
        row = dict(input_ids=tokens, labels=labels, positions=torch.arange(length))
        if image:
            tokens[4] = 120
            row["pixel_values"] = torch.ones(4, 1536) * 0.1
            row["image_grid_thw"] = torch.tensor([[1, 2, 2]])
        rows.append(row)
    return str(index), rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=0)
    parser.add_argument("--streams", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--loop-only", action="store_true")
    parser.add_argument("--production-kernels", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("Outputs must be inside this repository")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.autograd.set_multithreading_enabled(False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not args.production_kernels:
        install_fp32_reference_kernels()
    else:
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.kernels import (
            stabilize_gdn_output,
        )

        stabilize_gdn_output()
    dist.init_process_group("nccl")
    loop = _load_loop_module()
    spec = loop.model_registry(
        "debugmodel",
        seq_len=128,
        attn_backend="varlen",
        flash_attention_impl="FA2",
        loop_settings=loop.LoopSettings(
            start=2,
            end=4,
            counts=[1, 2],
            probabilities=[0.5, 0.5],
            eval_count=3,
            use_gate=True,
        ),
    )
    spec.model.vocab_size = (
        spec.model.tok_embeddings.num_embeddings
    ) = spec.model.lm_head.out_features = 128
    spec.model.freeze_vision = True
    spec.model.trainable_loop_only = args.loop_only
    if args.streams:
        spec.model.mhc = dict(
            start=2,
            end=4,
            streams=args.streams,
            iterations=20,
            fixed_identity=False,
            pre_scale=1.0,
        )
    parallel = ParallelismConfig(
        context_parallel_degree=args.cp,
        data_parallel_replicate_degree=args.replicate,
        data_parallel_shard_degree=-1,
    )
    spec.model.update_from_config(config=SimpleNamespace(parallelism=parallel))
    dims = ParallelDims.from_config(parallel, dist.get_world_size())
    training = TrainingConfig(
        dtype="float32",
        mixed_precision_param=("bfloat16" if args.production_kernels else "float32"),
        mixed_precision_reduce="float32",
        max_context_length=128,
        disable_cuda_graphs=True,
    )
    context = get_spmd_context(parallel_dims=dims, spmd_typechecking=False)
    torch.manual_seed(42)
    with torch.device("cuda"):
        model = spec.model.build()
    model.init_weights(buffer_device=None)
    with context():
        model = spec.parallelize_fn(
            model,
            parallel_dims=dims,
            parallelism=parallel,
            training=training,
            compile_config=None,
            ac_config=None,
            dump_folder=str(args.output.parent),
        )
    model._skip_lm_head = True
    model.train()
    engine = DPOTrainingEngine.__new__(DPOTrainingEngine)
    engine.model_spec, engine.model_config, engine.model_parts = (
        spec,
        spec.model,
        [model],
    )
    engine.parallel_dims, engine.device = (
        dims,
        torch.device("cuda", torch.cuda.current_device()),
    )
    engine.config = SimpleNamespace(training=training, parallelism=parallel)
    engine.output_dir, engine.train_context = str(args.output.parent), context
    engine.max_num_documents, engine.preprocess_inputs_kwargs = None, {}
    engine.loss_fn = DPOLoss.Config(num_chunks=args.chunks).build()
    engine.ntokens_seen = 0
    engine.reference = ReferenceRuntime(engine)
    dp = dims.get_optional_mesh("dp")
    dp_rank, dp_size = (dp.get_local_rank(), dp.size()) if dp is not None else (0, 1)
    loader = PreferenceDataLoader.__new__(PreferenceDataLoader)
    loader.token_budget, loader.max_length = (
        (512 if args.packing else 256),
        128,
    )
    loader.collator = MultiModalCollator.Config(build_mrope_positions=True).build(
        context=SimpleNamespace(
            tokenizer=SimpleNamespace(pad_id=0, image_id=120, video_id=121),
            num_tokens_per_microbatch=loader.token_budget,
            max_context_length=128,
        )
    )
    pairs = [
        pair(index, args.images and index == 0)
        for index in range(2)
        if index % dp_size == dp_rank
    ]
    batches = (
        [loader.collate(pairs)]
        if args.packing
        else [loader.collate([item]) for item in pairs]
    )
    physical = torch.tensor(len(batches), device="cuda")
    dist.all_reduce(physical, op=dist.ReduceOp.MAX)
    if args.replicate > 1:
        # A final zero-contribution accumulation flushes earlier HSDP gradients.
        physical += 1
    batches += [loader.collate([]) for _ in range(int(physical) - len(batches))]
    engine.num_accumulation_steps = len(batches)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0
    )
    frozen = {
        name: local_tensor(p).detach().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }
    reference_initial = {
        name: local_tensor(p).detach().clone()
        for name, p in engine.reference.model.named_parameters()
    }
    records, first_gradients = [], {}
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        model.set_training_step(step)
        totals = {}
        for index, batch in enumerate(batches):
            add_metrics(
                totals,
                engine.preference_microbatch(
                    batch,
                    global_pairs=2,
                    loop_count=model.loop_train_count,
                    backward=True,
                    accumulation_index=index,
                ),
            )
        metrics = reduce_metrics(totals, dims)
        if step == 0:
            torch.testing.assert_close(
                torch.tensor(metrics["loss"]),
                torch.tensor(2.0).log(),
                atol=1e-5,
                rtol=1e-4,
            )
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    gradient = (
                        parameter.grad.full_tensor()
                        if isinstance(parameter.grad, DTensor)
                        else parameter.grad
                    )
                    if dist.get_rank() == 0:
                        first_gradients[name] = gradient.detach().cpu()
        assert engine.loss_is_finite.item(), f"nonfinite loss step={step}"
        assert all(
            torch.isfinite(local_tensor(p.grad)).all()
            for p in model.parameters()
            if p.grad is not None
        ), step
        optimizer.step()
        records.append(
            dict(step=step + 1, loop_count=model.loop_train_count, **metrics)
        )
    for name, parameter in engine.reference.model.named_parameters():
        assert parameter.grad is None and torch.equal(
            local_tensor(parameter), reference_initial[name]
        ), (
            name,
            local_tensor(parameter).shape,
            reference_initial[name].shape,
            parameter.dtype,
        )
    assert all(
        torch.equal(local_tensor(dict(model.named_parameters())[name]), value)
        for name, value in frozen.items()
    )
    if dist.get_rank() == 0:
        payload = dict(
            records=records,
            gradients=first_gradients,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
        )
        if args.reference:
            baseline = torch.load(args.reference, weights_only=True)
            assert first_gradients.keys() == baseline["gradients"].keys()
            for name, gradient in first_gradients.items():
                torch.testing.assert_close(
                    gradient,
                    baseline["gradients"][name],
                    atol=1e-5,
                    rtol=1e-4,
                    msg=name,
                )
            for key in ("loss", "policy_chosen_logps", "policy_rejected_logps"):
                torch.testing.assert_close(
                    torch.tensor(records[0][key]),
                    torch.tensor(baseline["records"][0][key]),
                    atol=1e-5,
                    rtol=1e-4,
                    msg=key,
                )
        torch.save(payload, args.output)
        print(
            json.dumps(
                dict(
                    records=records,
                    peak_memory_bytes=payload["peak_memory_bytes"],
                    gradients=len(first_gradients),
                    comparison="passed" if args.reference else "baseline",
                )
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
