# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the real CLI using a registered debug Loop and local HF assets.

The debug architecture registration is confined to this test process. Production
model selection, templates, loading, trainers and checkpoint paths are unchanged.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT)]


def prepare(folder, processor_assets, streams):
    import torch
    from datasets import Dataset
    from safetensors.torch import save_file
    from torchtitan.experiments.loop_qwen35.config_registry import _load_loop_module
    from torchtitan.experiments.loop_qwen35.model_config import _native_architecture

    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    loop = _load_loop_module()
    settings = loop.LoopSettings(
        start=2,
        end=4,
        counts=[1, 2],
        probabilities=[0.5, 0.5],
        eval_count=3,
        use_gate=True,
    )
    spec = loop.model_registry(
        "debugmodel",
        seq_len=512,
        loop_settings=settings,
        attn_backend="varlen",
        flash_attention_impl="FA2",
    )
    if streams:
        spec.model.mhc = dict(
            start=2,
            end=4,
            streams=streams,
            iterations=20,
            fixed_identity=False,
            pre_scale=1.0,
        )
    from types import SimpleNamespace
    from torchtitan.config import ParallelismConfig

    spec.model.update_from_config(
        config=SimpleNamespace(parallelism=ParallelismConfig())
    )
    assets = folder / "model"
    assets.mkdir(exist_ok=True)
    for source in processor_assets.iterdir():
        if (
            source.is_file()
            and source.suffix in (".json", ".jinja", ".txt")
            and source.name not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copyfile(source, assets / source.name)
    raw = json.loads((processor_assets / "config.json").read_text())
    for name, value in _native_architecture(spec.model).items():
        target = raw
        components = name.split(".")
        for component in components[:-1]:
            target = target.setdefault(component, {})
        target[components[-1]] = value
    raw.update(
        loop_settings=settings.to_dict(),
        tie_word_embeddings=False,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    raw["text_config"]["max_position_embeddings"] = 32768
    (assets / "config.json").write_text(json.dumps(raw, indent=2))
    torch.manual_seed(42)
    model = spec.model.build()
    model.init_weights(buffer_device=None)
    weights = spec.state_dict_adapter(spec.model, str(assets)).to_hf(model.state_dict())
    save_file(
        {name: value.detach().contiguous() for name, value in weights.items()},
        str(assets / "model.safetensors"),
    )
    records = [
        json.loads(line)
        for line in (ROOT / "tests/experiments/loop_qwen35/fixtures/dpo_pairs.jsonl")
        .read_text()
        .splitlines()
    ]
    Dataset.from_list(records).save_to_disk(str(folder / "pairs"))
    print(
        json.dumps(
            dict(
                assets=str(assets),
                pairs=str(folder / "pairs"),
                parameters=sum(p.numel() for p in model.parameters()),
            )
        )
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--processor-assets", type=Path, required=True)
        parser.add_argument("--streams", type=int, choices=(0, 2, 4), default=0)
        args = parser.parse_args()
        prepare(args.output, args.processor_assets, args.streams)
        return
    import torchtitan.experiments.loop_qwen35.model_config as model_config

    model_config.DENSE_FLAVORS = (*model_config.DENSE_FLAVORS, "debugmodel")
    factory, _ = model_config.qwen3_5_configs["debugmodel"]
    model_config.qwen3_5_configs["debugmodel"] = (factory, 32768)
    import torch
    import torch.distributed as dist
    import torchtitan.experiments.loop_qwen35.dpo.trainer as dpo_trainer
    from torchtitan.experiments.loop_qwen35.dpo.checkpoint import (
        digest_state,
        snapshot_state,
    )
    from torchtitan.experiments.loop_qwen35.dpo.reference import local_tensor

    class VerifiedTrainer(dpo_trainer.DPOTrainer):
        def on_checkpoint_loaded(self, resumed):
            super().on_checkpoint_loaded(resumed)
            assert not any(
                name == "swift" or name.startswith("swift.") for name in sys.modules
            )
            self.reference_digest = digest_state(
                snapshot_state(self.engine.reference.parts)
            )
            self.policy_initial = {
                name: local_tensor(p).detach().clone()
                for name, p in self.engine.model_parts[0].named_parameters()
                if name.startswith("layers.2.")
            }
            self.frozen_initial = {
                name: local_tensor(p).detach().clone()
                for name, p in self.engine.model_parts[0].named_parameters()
                if not p.requires_grad
            }
            if not resumed:
                assert self.reference_digest == digest_state(
                    snapshot_state(self.engine.model_parts)
                )

        def train_step(self, iterator):
            super().train_step(iterator)
            assert all(
                p.grad is None and not p.requires_grad
                for p in self.engine.reference.model.parameters()
            )
            if self.engine.num_completed_steps == self.config.training.steps:
                assert self.reference_digest == digest_state(
                    snapshot_state(self.engine.reference.parts)
                )
                parameters = dict(self.engine.model_parts[0].named_parameters())
                assert any(
                    not torch.equal(initial, local_tensor(parameters[name]))
                    for name, initial in self.policy_initial.items()
                )
                assert all(
                    torch.equal(initial, local_tensor(parameters[name]))
                    for name, initial in self.frozen_initial.items()
                )
                report = dict(
                    step=self.engine.num_completed_steps,
                    reference_unchanged=True,
                    policy_updated=True,
                    frozen_unchanged=True,
                    reference_digest=self.reference_digest,
                    peak_memory_bytes=torch.cuda.max_memory_allocated(),
                )
                (
                    Path(self.config.dump_folder)
                    / f"verification-rank-{dist.get_rank()}.json"
                ).write_text(json.dumps(report, indent=2))

    dpo_trainer.DPOTrainer = VerifiedTrainer
    from torchtitan.experiments.loop_qwen35.train import main as train

    train()


if __name__ == "__main__":
    main()
