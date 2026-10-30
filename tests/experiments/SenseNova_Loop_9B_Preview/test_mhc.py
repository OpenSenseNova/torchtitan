# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchtitan.experiments.SenseNova_Loop_9B_Preview.mhc.model import (
    collapse_streams,
    expand_streams,
    MHCBlock,
    MHCRouter,
    sinkhorn,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.mhc.schedule import (
    factor,
    MHCSchedulers,
)
from torchtitan.models.qwen3_5.model import Qwen35TransformerBlock
from torchtitan.protocols.module import Module

ROOT = Path(__file__).resolve().parents[3]


def recipe():
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.models.qwen35_loop import (
        LoopSettings,
    )

    return dict(
        model_flavor="4B",
        model_path="unused",
        loop_settings=LoopSettings(
            start=4, end=20, counts=[2], probabilities=[1.0], use_gate=False
        ).to_dict(),
    )


class Branch(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 8

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.dim, config.dim)

    def forward(self, x, *args, **kwargs):
        return self.linear(x).tanh()


class Norm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


def blocks():
    args = dict(
        attention=Branch.Config(),
        feed_forward=Branch.Config(),
        attention_norm=Norm.Config(),
        ffn_norm=Norm.Config(),
    )
    base = Qwen35TransformerBlock.Config(**args).build()
    model = MHCBlock.Config(**args, mhc_dim=8).build()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert len(missing) == 4 and not unexpected
    return base, model


class MHCTest(unittest.TestCase):
    def test_imports_use_current_checkout(self):
        import torchtitan
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
            _load_loop_module,
        )

        for mod in (torchtitan, _load_loop_module()):
            self.assertTrue(
                Path(mod.__file__).resolve().is_relative_to(ROOT), mod.__file__
            )

    def test_sinkhorn_finite_and_constraints(self):
        z = (torch.randn(12, 4, 4) * 2).requires_grad_()
        m = sinkhorn(z, 100)
        torch.testing.assert_close(m.sum(-1), torch.ones(12, 4), atol=2e-5, rtol=0)
        torch.testing.assert_close(m.sum(-2), torch.ones(12, 4), atol=2e-5, rtol=0)
        m.square().sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.isfinite(sinkhorn(torch.randn(2, 4, 4) * 1000)).all())

    def test_initial_output_and_input_gradient(self):
        torch.manual_seed(42)
        base, mhc = blocks()
        x = torch.randn(16, 8, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a = base(x, None)
        b = collapse_streams(mhc(expand_streams(y, 2), None), 2)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(mhc.mhc_attn.weight.grad).all())
        self.assertGreater(mhc.mhc_attn.weight.grad.abs().sum().item(), 0)
        # Identical streams need not give a nonzero residual-mixing gradient at init.
        self.assertFalse(
            torch.equal(mhc.mhc_attn.bias.grad[2:3], mhc.mhc_attn.bias.grad[3:4])
        )

    def test_checkpoint_recompute_and_loop_sharing(self):
        torch.manual_seed(9)
        _, model = blocks()
        other = copy.deepcopy(model)
        x = torch.randn(8, 8, requires_grad=True)
        z = x.detach().clone().requires_grad_()

        def run(m, t, recompute):
            for _ in range(2):
                t = expand_streams(t, 2)
                t = (
                    checkpoint(m, t, None, use_reentrant=False)
                    if recompute
                    else m(t, None)
                )
                t = collapse_streams(t, 2)
            return t

        a = run(model, x, False)
        b = run(other, z, True)
        torch.testing.assert_close(a, b)
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_cp_token_partition_equivalence(self):
        router = MHCRouter(8)
        x = torch.randn(12, 2, 8)
        with torch.no_grad():
            router.weight.normal_(std=0.01)
        whole = router(x)
        left, right = router(x[:6]), router(x[6:])
        for a, b, c in zip(whole, left, right):
            torch.testing.assert_close(a, torch.cat((b, c)))

    def test_native_state_round_trip(self):
        _, a = blocks()
        _, b = blocks()
        opt = torch.optim.AdamW(a.parameters(), lr=1e-3)
        x = torch.randn(12, 8)
        loss = collapse_streams(a(expand_streams(x, 2), None), 2).square().mean()
        loss.backward()
        opt.step()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "weights.pt"
            torch.save(a.state_dict(), p)
            b.load_state_dict(torch.load(p, weights_only=True))
        torch.testing.assert_close(
            a(expand_streams(x, 2), None), b(expand_streams(x, 2), None)
        )

    def test_schedule_groups_and_resume(self):
        def build():
            a, b = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
            opt = torch.optim.AdamW(
                [
                    {
                        "params": [a],
                        "lr": 1e-4,
                        "param_names": ["layers.4.mhc_attn.weight"],
                    },
                    {
                        "params": [b],
                        "lr": 1e-5,
                        "param_names": ["layers.4.attn.weight"],
                    },
                ]
            )
            sched = MHCSchedulers([opt], 100, 0, 10, 0.0)
            return opt, sched

        opt, sched = build()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-5)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 1e-5)
        for _ in range(15):
            opt.step()
            sched.step()
        clone, cs = build()
        clone.load_state_dict(opt.state_dict())
        cs.load_state_dict(sched.state_dict())
        self.assertEqual(
            [g["lr"] for g in opt.param_groups], [g["lr"] for g in clone.param_groups]
        )
        opt.step()
        sched.step()
        clone.step()
        cs.step()
        self.assertEqual(sched.get_metrics(), cs.get_metrics())
        self.assertEqual(factor(100, 100, 10, 0), 0.0)

    def test_identity_ablation(self):
        router = MHCRouter(8, fixed_identity=True)
        _, _, mix = router(torch.randn(5, 2, 8))
        torch.testing.assert_close(mix, torch.eye(2).expand(5, 2, 2))

    def test_actual_qwen_config_and_meta_build(self):
        from torchtitan.config import ParallelismConfig
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
            qwen35_loop,
        )

        resolved = recipe()
        cfg = qwen35_loop(resolved, seq_len=8192)
        cfg.parallelism = ParallelismConfig(
            tensor_parallel_degree=1, context_parallel_degree=2
        )
        cfg.model_spec.model.mhc = dict(
            start=4, end=8, streams=2, iterations=20, fixed_identity=False
        )
        cfg.model_spec.model.update_from_config(config=cfg)
        self.assertIsInstance(cfg.model_spec.model.layers[4], MHCBlock.Config)
        with torch.device("meta"):
            model = cfg.model_spec.model.build()
        self.assertIsInstance(model.layers["4"], MHCBlock)
        self.assertNotIsInstance(model.layers["3"], MHCBlock)
        self.assertEqual(
            len([n for n, p in model.named_parameters() if ".mhc_" in n]), 16
        )

    def test_hf_import_whitelists_only_new_router_parameters(self):
        from types import SimpleNamespace

        from safetensors.torch import save_file
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.checkpoint import (
            LoopCheckpoint,
        )

        _, block = blocks()
        part = nn.ModuleDict(
            {"layers": nn.ModuleDict({"4": checkpoint_wrapper(block)})}
        )
        part.config = SimpleNamespace(layers=[None] * 5)
        original = {k: v.clone() for k, v in part.state_dict().items()}
        source = {
            k: torch.full_like(v, 0.125)
            for k, v in original.items()
            if ".mhc_" not in k
        }
        source["lm_head.weight"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.safetensors"
            save_file(source, str(path))
            loader = object.__new__(LoopCheckpoint)
            loader.config = LoopCheckpoint.Config(initial_load_path=d)
            loader.parts = [part]
            loader.adapter = SimpleNamespace(
                hf_language_model_prefix="model",
                from_hf=lambda weights: {
                    k: v for k, v in weights.items() if k != "lm_head.weight"
                },
            )
            loader.engine = SimpleNamespace(
                device="cpu", parallel_dims=SimpleNamespace(tp=1)
            )
            loader._load_hf()
            for k, v in part.state_dict().items():
                torch.testing.assert_close(
                    v, original[k] if ".mhc_" in k else torch.full_like(v, 0.125)
                )
            source.pop(next(k for k in source if k != "lm_head.weight"))
            save_file(source, str(path))
            with self.assertRaisesRegex(KeyError, "missing required model tensor"):
                loader._load_hf()

    def test_real_optimizer_group_partition(self):
        from torchtitan.components.optimizer import default_adamw, ParamGroupConfig

        _, block = blocks()
        model = nn.ModuleDict({"layers": nn.ModuleDict({"4": block})})
        cfg = default_adamw(lr=1e-5)
        cfg.implementation = "for-loop"
        cfg.param_groups.insert(
            0,
            ParamGroupConfig(
                pattern=r"\.mhc_(attn|ffn)\.",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
            ),
        )
        optimizers = cfg.build(model_parts=[model])
        sched = MHCSchedulers.Config(
            total_steps=100, warmup_steps=0, router_warmup_steps=10
        ).build(optimizers=optimizers, training_steps=100)
        self.assertEqual(set(sched.get_metrics()), {"lr/router", "lr/backbone"})
        all_params = [
            p for opt in optimizers for g in opt.param_groups for p in g["params"]
        ]
        self.assertEqual(len(all_params), len(list(model.parameters())))
        self.assertEqual(len({id(p) for p in all_params}), len(all_params))

    def test_bfloat16_initial_function_preservation(self):
        base, mhc = blocks()
        base.to(torch.bfloat16)
        mhc.to(torch.bfloat16)
        mhc.mhc_attn.float()
        mhc.mhc_ffn.float()
        x = torch.randn(16, 8, dtype=torch.bfloat16)
        a = base(x, None)
        b = collapse_streams(mhc(expand_streams(x, 2), None), 2)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_adapter_router_round_trip(self):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.config_registry import (
            _load_loop_module,
            qwen35_loop,
        )

        resolved = recipe()
        cfg = qwen35_loop(resolved, seq_len=8192)
        adapter = _load_loop_module().Qwen35LoopStateDictAdapter(
            cfg.model_spec.model, None
        )
        weights = {
            "layers.4.mhc_attn.weight": torch.randn(8, 16),
            "layers.4.mhc_ffn.bias": torch.randn(8),
            "lm_head.weight": torch.randn(16, 8),
        }
        restored = adapter.from_hf(adapter.to_hf(weights))
        self.assertEqual(set(restored), set(weights))
        for key in weights:
            torch.testing.assert_close(weights[key], restored[key])


if __name__ == "__main__":
    unittest.main()
