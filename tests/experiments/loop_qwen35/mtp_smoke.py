# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Audit actual HF MTP training and record per-step gradient evidence.

Accepts the same arguments as run_regression.py. Run under torchrun.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from torchtitan.experiments.loop_qwen35 import trainer as sft

    def fingerprints(parameters):
        result = {}
        for name, value in parameters:
            value = value.to_local() if isinstance(value, DTensor) else value
            raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
            result[name] = hashlib.sha256(memoryview(raw)).hexdigest()
        return result

    class VerifiedTrainer(sft.SFTTrainer):
        def on_checkpoint_loaded(self, resumed):
            super().on_checkpoint_loaded(resumed)
            assert self.engine.mtp_settings
            self.initial = fingerprints(self.engine.model_parts[0].named_parameters())
            self.frozen_names = {
                name
                for name, p in self.engine.model_parts[0].named_parameters()
                if not p.requires_grad
            }
            self.resumed = resumed

        def train_step(self, iterator):
            super().train_step(iterator)
            parameters = dict(self.engine.model_parts[0].named_parameters())
            assert all(parameters[name].grad is None for name in self.frozen_names)
            metrics = self.engine.last_mtp_step_metrics
            assert metrics["tokens/mtp_valid"] > 0
            assert metrics["mtp/grad_norm"] > 0
            if self.engine.num_completed_steps != self.config.training.steps:
                return
            final = fingerprints(parameters.items())
            changed = {
                name for name, digest in self.initial.items() if final[name] != digest
            }
            assert not changed.intersection(self.frozen_names)
            assert any(name.startswith("mtp_layers.") for name in changed)
            backbone_changed = any(
                not name.startswith("mtp_layers.") for name in changed
            )
            mode = self.engine.mtp_settings["training_mode"]
            assert backbone_changed == (mode == "joint")
            report = dict(
                step=self.engine.num_completed_steps,
                mode=mode,
                resumed=self.resumed,
                frozen_unchanged=True,
                mtp_updated=True,
                backbone_updated=backbone_changed,
                changed_parameters=sorted(changed),
                final_fingerprints=final,
                metrics=metrics,
            )
            (
                Path(self.config.dump_folder)
                / f"verification-rank-{dist.get_rank()}.json"
            ).write_text(json.dumps(report, indent=2))

    sft.SFTTrainer = VerifiedTrainer
    from run_regression import main as train

    train()


if __name__ == "__main__":
    main()
