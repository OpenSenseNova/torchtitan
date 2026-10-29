# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU regression for loop execution, scheduling and packed CP metadata."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torchtitan.experiments.loop_qwen35.config_registry import _load_loop_module

ROOT = Path(__file__).resolve().parents[3]

loop = _load_loop_module()


class AffineLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0 + index / 10))
        self.bias = index / 7

    def forward(self, x, *args, **kwargs):
        return self.weight * x + self.bias


def affine_model(layers):
    model = loop.Qwen35LoopModel.__new__(loop.Qwen35LoopModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(layers=list(range(8)), mhc=None)
    model.layers = nn.ModuleDict({str(i): layer for i, layer in enumerate(layers)})
    model.loop_settings = loop.LoopSettings(start=2, end=4, use_gate=False)
    model.loop_train_count = 2
    model.tok_embeddings = model.norm = model.lm_head = None
    model._skip_lm_head = True
    return model


class LoopSemanticsTest(unittest.TestCase):
    def test_pipeline_parallelism_is_explicitly_unsupported(self):
        from torchtitan.config import ParallelismConfig

        spec = loop.model_registry(
            "debugmodel", loop_settings=loop.LoopSettings(start=2, end=4)
        )
        config = SimpleNamespace(
            parallelism=ParallelismConfig(pipeline_parallel_degree=2)
        )
        self.assertIsNone(spec.pipelining_fn)
        with self.assertRaisesRegex(ValueError, "supports PP=1"):
            spec.model.update_from_config(config=config)

    def test_prepare_step_uses_restored_step_for_all_microbatches(self):
        from torchtitan.experiments.loop_qwen35.engine import (
            LoopTrainingEngine,
            TrainingEngine,
        )

        model = affine_model([AffineLayer(i) for i in range(8)])
        model.loop_settings = loop.LoopSettings(start=2, end=4)
        engine = LoopTrainingEngine.__new__(LoopTrainingEngine)
        engine.model_parts = [model]
        engine.num_completed_steps = 7
        with patch.object(TrainingEngine, "prepare_step", return_value=123) as prepare:
            for step in (7, 8):
                engine.num_completed_steps = step
                result = engine.prepare_step(123, num_accumulation_steps=4)
                self.assertEqual(result, 123)
                self.assertEqual(
                    model.loop_train_count, model.loop_settings.count_for_step(step)
                )
                prepare.assert_called_with(123, num_accumulation_steps=4)

    def test_gdn_all_to_all_rejects_cp_that_splits_a_key_head(self):
        from torchtitan.config import ParallelismConfig

        spec = loop.model_registry(
            "debugmodel",
            seq_len=512,
            attn_backend="varlen",
            loop_settings=loop.LoopSettings(
                start=2, end=4, counts=[2], probabilities=[1.0]
            ),
        )
        spec.model.cp_attention_backend = "all_gather"
        # Debug GDN has two key heads: TP2 + CP2 would cut a head in half.
        config = SimpleNamespace(
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2, context_parallel_degree=2
            )
        )
        with self.assertRaisesRegex(ValueError, r"GDN all-to-all requires TP\*CP"):
            spec.model.update_from_config(config=config)

    def test_cp_preserves_sp_and_all_module_layouts_validate(self):
        import dataclasses

        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed.spmd_types import spmd_validate_redistributions
        from torchtitan.protocols.sharding import ShardingConfig

        def validate_tree(value):
            if isinstance(value, ShardingConfig):
                spmd_validate_redistributions(value)
            elif dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    validate_tree(getattr(value, field.name))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    validate_tree(item)
            elif isinstance(value, dict):
                for item in value.values():
                    validate_tree(item)

        for backend, tp in (("all_gather", 4), ("ulysses", 2)):
            for enable_sp in (True, False):
                with self.subTest(backend=backend, enable_sp=enable_sp):
                    spec = loop.model_registry("4B", seq_len=512, attn_backend="varlen")
                    spec.model.cp_attention_backend = backend
                    parallelism = ParallelismConfig(
                        tensor_parallel_degree=tp,
                        context_parallel_degree=2,
                        enable_sequence_parallel=enable_sp,
                    )
                    spec.model.update_from_config(
                        config=SimpleNamespace(parallelism=parallelism)
                    )
                    self.assertEqual(parallelism.enable_sequence_parallel, enable_sp)
                    self.assertIsNotNone(spec.model.input_shard)
                    validate_tree(spec.model)

    def test_cp_flash_metadata_keeps_causal_prefix_across_rank_boundary(self):
        from torchtitan.models.common.attention import VarlenMetadata
        from torchtitan.models.common.cp_attention import make_cp_varlen_metadata

        cu = torch.tensor([0, 63, 64, 258, 263, 512], dtype=torch.int32)
        global_metadata = VarlenMetadata(cu, cu, 249, 249)
        local = make_cp_varlen_metadata(global_metadata, rank=1, size=2, num_tokens=512)
        self.assertEqual(local.local.cu_seq_q.tolist(), [0, 2, 7, 256])
        self.assertEqual(local.local.cu_seq_k.tolist(), [0, 194, 199, 448])
        self.assertEqual((local.kv_start, local.kv_end), (64, 512))
        self.assertEqual(local.local.max_q, 249)
        first = make_cp_varlen_metadata(global_metadata, rank=0, size=2, num_tokens=512)
        self.assertEqual(first.local.cu_seq_q.tolist(), [0, 63, 64, 256])
        self.assertEqual(first.local.cu_seq_q.tolist(), first.local.cu_seq_k.tolist())

    def test_packing_preserves_singleton_documents_and_padding_runs(self):
        model = loop.Qwen35LoopModel.__new__(loop.Qwen35LoopModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(first_attention=None)
        positions = torch.tensor([0, 1, 2, 0, 0, 1, 2])
        metadata = model.get_attention_masks(positions)["deltanet"]
        self.assertEqual(metadata.cu_seq_q.tolist(), [0, 3, 4, 7])
        positions = torch.tensor([0, 1, 0, 0, 0, 0, 1])
        padding = torch.tensor([False, False, True, True, True, False, False])
        metadata = model.get_attention_masks(positions, padding_mask=padding)[
            "deltanet"
        ]
        self.assertEqual(metadata.cu_seq_q.tolist(), [0, 2, 5, 7])

    def test_arrow_image_metadata_reaches_vision_scatter(self):
        from torchtitan.experiments.loop_qwen35.data import ArrowDataLoader
        from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator

        tokenizer = SimpleNamespace(pad_id=0, image_id=42, video_id=43)
        loader = ArrowDataLoader.__new__(ArrowDataLoader)
        loader.config = SimpleNamespace(
            packing=False, seed=42, packing_strategy="greedy"
        )
        loader.dataset = [{"messages": []}]
        loader.dp_rank, loader.dp_world_size = 0, 1
        loader.epoch = loader.cursor = loader.skipped = 0
        loader.max_length = loader.token_budget = 4
        loader._set_order()
        encoded = {
            "input_ids": [5, 42, 6],
            "labels": [-100, -100, 6],
            "pixel_values": torch.zeros(4, 1536),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        loader.template = SimpleNamespace(encode=lambda sample: encoded)
        loader.collator = MultiModalCollator.Config(build_mrope_positions=True).build(
            context=SimpleNamespace(
                tokenizer=tokenizer, num_tokens_per_microbatch=4, max_context_length=4
            )
        )
        batch = next(loader)
        kwargs = batch.to_input_dict("cpu")
        self.assertEqual(kwargs["special_tokens"], {"image_id": 42, "video_id": 43})
        self.assertEqual(tuple(kwargs["mrope_positions"].shape), (4, 3))

        model = loop.Qwen35LoopModel.__new__(loop.Qwen35LoopModel)
        nn.Module.__init__(model)
        model.tok_embeddings = nn.Embedding(64, 4)
        model.vision_encoder = nn.Identity()
        model.vision_encoder.spatial_merge_unit = 4
        vision = torch.full((1, 4), 7.0)
        with patch.object(
            model, "_get_vision_embeds", return_value=(vision, torch.tensor([1]))
        ):
            result = model._prepare_multimodal_embeds(
                batch.input,
                pixel_values=kwargs["pixel_values"],
                grid_thw=kwargs["grid_thw"],
                pixel_values_videos=None,
                grid_thw_videos=None,
                special_tokens=kwargs["special_tokens"],
            )
        torch.testing.assert_close(result[1], vision[0])
        torch.testing.assert_close(
            result[[0, 2, 3]], model.tok_embeddings(batch.input[[0, 2, 3]])
        )

    def test_forward_and_gradients_match_explicit_loop(self):
        layers = [AffineLayer(i) for i in range(8)]
        x = torch.tensor([0.2, -0.1], requires_grad=True)
        reference = x
        for i in [0, 1, 2, 3, 2, 3, 4, 5, 6, 7]:
            reference = layers[i](reference)
        params = [x] + [layer.weight for layer in layers]
        expected_grads = torch.autograd.grad(reference.sum(), params)
        actual = affine_model(layers)(x)
        actual_grads = torch.autograd.grad(actual.sum(), params)
        torch.testing.assert_close(actual, reference)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
