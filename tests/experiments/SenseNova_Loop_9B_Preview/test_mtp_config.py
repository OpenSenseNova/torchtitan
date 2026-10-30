# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit mode and source-format contracts for pretrained Loop MTP."""

import itertools
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Select the repository under test.

from torchtitan.experiments.SenseNova_Loop_9B_Preview.model_config import (
    resolve_mtp_config,
    validate_mtp_source_config,
)  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


class MTPConfigTest(unittest.TestCase):
    def test_mtp_cli_and_long_context_parse(self):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.config import (
            parse_args,
        )

        for length, chunks in ((512, 4), (262144, 256)):
            with patch.dict("os.environ", {"WORLD_SIZE": "2"}):
                args = parse_args(
                    [
                        "--model",
                        "/model",
                        "--dataset",
                        "/dataset",
                        "--output-dir",
                        "/output",
                        "--mtp-num-layers",
                        "1",
                        "--cp",
                        "2",
                        "--max-length",
                        str(length),
                        "--loss-chunks",
                        str(chunks),
                        "--resume",
                        "false",
                    ]
                )
            self.assertEqual(args.max_length, length)
            self.assertEqual(args.mtp_training_mode, "mtp_only")

    def test_disabled_does_not_enable_from_hf(self):
        config = resolve_mtp_config()
        self.assertFalse(config["enabled"])
        validate_mtp_source_config(
            {"text_config": {"mtp_num_hidden_layers": 1}}, config, model_flavor="4B"
        )

    def test_mode_defaults(self):
        frozen = resolve_mtp_config(num_layers=1)
        joint = resolve_mtp_config(num_layers=1, training_mode="joint")
        self.assertEqual(frozen["loss_scaling_factor"], 1.0)
        self.assertEqual(joint["loss_scaling_factor"], 0.1)
        self.assertTrue(frozen["detach_backbone"])
        self.assertFalse(joint["detach_backbone"])
        self.assertFalse(frozen["automatic_unfreezing"])
        self.assertEqual(json.loads(json.dumps(frozen)), frozen)

    def test_parallel_head_and_local_token_constraints(self):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.model_config import (
            validate_parallelism,
        )

        base = {
            f"text_config.{name}": value
            for name, value in {
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "vocab_size": 248320,
            }.items()
        }
        resolved = {"base_model": base, "mtp": resolve_mtp_config(num_layers=1)}
        for tp, cp in ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1)):
            for chunks in (0, 1, 4):
                validate_parallelism(
                    resolved,
                    tp=tp,
                    cp=cp,
                    tokens_per_microbatch=512,
                    loss_chunks=chunks,
                )
        for values in (
            {"tp": 0, "cp": 1},
            {"tp": 3, "cp": 1},
            {"tp": 2, "cp": 16},
            {"tp": 2, "cp": 2, "tokens_per_microbatch": 14, "loss_chunks": 4},
            {"tp": 2, "cp": 2, "tokens_per_microbatch": 0},
            {"tp": 2, "cp": 2, "loss_chunks": -1},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_parallelism(resolved, **values)
        base["text_config.vocab_size"] += 1
        with self.assertRaisesRegex(ValueError, "shared head"):
            validate_parallelism(resolved, tp=2, cp=1)

    def test_real_9b_mtp_layouts_and_extra_container_are_isolated(self):
        import dataclasses
        import torch
        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed.spmd_types import spmd_validate_redistributions
        from torchtitan.models.common.cp_attention import (
            KVAllGatherCPVarlenInnerAttention,
            UlyssesCPVarlenInnerAttention,
        )
        from torchtitan.protocols.sharding import ShardingConfig
        from types import SimpleNamespace

        from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
            _load_loop_module,
        )
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.models.qwen35_mtp import (
            build_mtp_config,
        )

        def validate(value):
            if isinstance(value, ShardingConfig):
                spmd_validate_redistributions(value)
            elif dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    validate(getattr(value, field.name))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    validate(item)
            elif isinstance(value, dict):
                for item in value.values():
                    validate(item)

        loop = _load_loop_module()
        for sp, backend in itertools.product((False, True), ("all_gather", "ulysses")):
            for cp in (1, 2):
                with self.subTest(sp=sp, cp=cp, backend=backend):
                    config = loop.model_registry(
                        "9B",
                        seq_len=512,
                        attn_backend="varlen",
                        flash_attention_impl="FA2",
                        loop_settings=loop.LoopSettings(start=4, end=28),
                    ).model
                    config.cp_attention_backend = backend
                    original = tuple(id(layer) for layer in config.layers)
                    config.mtp = resolve_mtp_config(num_layers=1)
                    config.mtp_layers = [build_mtp_config(config)]
                    self.assertIsNot(
                        config.mtp_layers[0].attention, config.layers[3].attention
                    )
                    config.update_from_config(
                        config=SimpleNamespace(
                            parallelism=ParallelismConfig(
                                tensor_parallel_degree=2,
                                context_parallel_degree=cp,
                                enable_sequence_parallel=sp,
                                activation_reduce_dtype="float32",
                            )
                        )
                    )
                    validate(config)
                    self.assertEqual(
                        tuple(id(layer) for layer in config.layers), original
                    )
                    self.assertEqual(len(config.layers), 32)
                    if cp == 2:
                        self.assertIsInstance(
                            config.mtp_layers[0].attention.inner_attention,
                            (
                                KVAllGatherCPVarlenInnerAttention
                                if backend == "all_gather"
                                else UlyssesCPVarlenInnerAttention
                            ).Config,
                        )
                        if backend == "all_gather":
                            self.assertEqual(
                                config.mtp_layers[
                                    0
                                ].attention.inner_attention.reduce_dtype,
                                "float32",
                            )
                    with torch.device("meta"):
                        model = config.build()
                    self.assertEqual(
                        sum(p.requires_grad for p in model.parameters()), 14
                    )
                    for name, parameter in model.mtp_layers[0].named_parameters():
                        module_name, _, key = name.rpartition(".")
                        module = model.mtp_layers[0].get_submodule(module_name)
                        self.assertIn(key, module._sharding_config.state_shardings)

    def test_coefficients_and_depths(self):
        for depth in (-1, 2, True, 1.0):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                resolve_mtp_config(num_layers=depth)
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            for mode in ("mtp_only", "joint"):
                with (
                    self.subTest(value=value, mode=mode),
                    self.assertRaises(ValueError),
                ):
                    resolve_mtp_config(
                        num_layers=1, training_mode=mode, loss_scaling_factor=value
                    )
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_mtp_config(num_layers=1, loss_scaling_factor=0.0)
        self.assertEqual(
            resolve_mtp_config(
                num_layers=1, training_mode="joint", loss_scaling_factor=0.0
            )["loss_scaling_factor"],
            0.0,
        )
        self.assertEqual(
            resolve_mtp_config(num_layers=1, loss_scaling_factor=0.3)[
                "loss_scaling_factor"
            ],
            0.3,
        )

    def test_freeze_conflict(self):
        with self.assertRaisesRegex(ValueError, "freeze_vision"):
            resolve_mtp_config(num_layers=1, freeze_vision=False)
        self.assertTrue(
            resolve_mtp_config(
                num_layers=1, training_mode="joint", freeze_vision=False
            )["enabled"]
        )

    def test_source_contract(self):
        mtp = resolve_mtp_config(num_layers=1)
        config = {
            "text_config": {
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": False,
            }
        }
        validate_mtp_source_config(config, mtp, model_flavor="9B")
        for bad in (
            {"text_config": {}},
            {"text_config": {"mtp_num_hidden_layers": 2}},
            {**config, "quantization_config": {"bits": 4}},
            {
                "text_config": {
                    "mtp_num_hidden_layers": 1,
                    "mtp_use_dedicated_embeddings": True,
                }
            },
        ):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                validate_mtp_source_config(bad, mtp, model_flavor="9B")
        with self.assertRaisesRegex(ValueError, "9B"):
            validate_mtp_source_config(config, mtp, model_flavor="4B")

    def test_cli_validation(self):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.train import (
            parse_args,
        )

        base = [
            "train",
            "--model",
            "/model",
            "--cached-dataset",
            "/data",
            "--output-dir",
            "/output",
            "--max-length",
            "512",
            "--loss-chunks",
            "4",
        ]
        with patch.object(sys, "argv", base):
            self.assertEqual(parse_args().mtp_num_layers, 0)
        with patch.object(sys, "argv", base + ["--mtp-num-layers", "1"]):
            self.assertEqual(parse_args().mtp_training_mode, "mtp_only")
        for options in (
            ["--mtp-num-layers", "2"],
            ["--mtp-num-layers", "1", "--mtp-loss-scaling-factor", "0"],
            ["--mtp-num-layers", "1", "--freeze-vision", "false"],
            ["--mtp-num-layers", "1", "--pp", "2"],
        ):
            with (
                self.subTest(options=options),
                patch.object(sys, "argv", base + options),
            ):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_9b_registry_modes_preserve_head_and_frozen_set(self):
        import torch

        from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
            _load_loop_module,
            qwen35_loop,
        )
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.loss import (
            LoopMTPLoss,
        )
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.models.qwen35_mtp import (
            parameter_manifest,
        )

        loop = _load_loop_module().LoopSettings(
            start=4, end=28, counts=[2], probabilities=[1.0], use_gate=False
        )
        for depth, mode in ((0, "mtp_only"), (1, "mtp_only"), (1, "joint")):
            with self.subTest(depth=depth, mode=mode):
                settings = resolve_mtp_config(num_layers=depth, training_mode=mode)
                config = qwen35_loop(
                    {
                        "model_flavor": "9B",
                        "model_path": "/unused",
                        "loop_settings": loop.to_dict(),
                        "mtp": settings,
                    },
                    seq_len=512,
                )
                with torch.device("meta"):
                    model = config.model_spec.model.build()
                self.assertIsNot(model.lm_head.weight, model.tok_embeddings.weight)
                self.assertEqual(len(model.layers), 32)
                manifest = parameter_manifest(model)
                if depth:
                    self.assertIsInstance(config.loss, LoopMTPLoss.Config)
                    self.assertEqual(len(model.mtp_layers[0].state_dict()), 15)
                    if mode == "mtp_only":
                        self.assertEqual(len(manifest["trainable_parameters"]), 14)
                        self.assertTrue(
                            all(
                                name.startswith("mtp_layers.")
                                for name in manifest["trainable_parameters"]
                            )
                        )
                        self.assertFalse(
                            any(
                                name.startswith("mtp_layers.")
                                for name in manifest["frozen_parameters"]
                            )
                        )
                    else:
                        self.assertIn(
                            "tok_embeddings.weight", manifest["trainable_parameters"]
                        )
                else:
                    self.assertIsNone(model.mtp_layers)
                    self.assertIn(
                        "tok_embeddings.weight", manifest["trainable_parameters"]
                    )


if __name__ == "__main__":
    unittest.main()
