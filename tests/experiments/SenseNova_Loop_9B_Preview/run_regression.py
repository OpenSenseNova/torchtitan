# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Run the production entry with per-step numerical evidence.

Runs reject Swift imports and record deterministic samples from every gradient.
"""

import argparse
import hashlib
import importlib
import importlib.abc
import json
import os
import sys
from pathlib import Path


class RejectSwift(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "swift" or fullname.startswith("swift."):
            raise ImportError(f"Standalone regression forbids Swift: {fullname}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    training_args = args.training_args
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    sys.meta_path.insert(0, RejectSwift())
    import torch
    from torch.distributed.tensor import DTensor
    from torchtitan.experiments.SenseNova_Loop_9B_Preview import train as entry
    from torchtitan.experiments.SenseNova_Loop_9B_Preview import (
        trainer as validation,
    )

    training_args += ["--num-layers", str(args.num_layers)]
    parsed = entry.parse_args(training_args)
    output = Path(parsed.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / f"numerics-rank-{os.environ['RANK']}.jsonl"
    original_step = validation.SFTTrainingEngine.optimizer_step

    def audited_step(engine):
        gradients = {}
        for part in engine.model_parts:
            for name, parameter in part.named_parameters():
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if isinstance(grad, DTensor):
                    grad = grad.to_local()
                # A deterministic sample from every tensor catches model/router
                # changes without retaining full 9B gradient copies on the host.
                flat = grad.detach().reshape(-1)
                stride = max(1, flat.numel() // 256)
                sample = flat[::stride][:256].contiguous().cpu()
                gradients[name] = hashlib.sha256(
                    sample.view(torch.uint8).numpy().tobytes()
                ).hexdigest()
        result = original_step(engine)
        grad_norm = result.item() if hasattr(result, "item") else result
        record = dict(
            step=engine.num_completed_steps,
            gradient_samples=gradients,
            grad_norm=grad_norm,
        )
        with evidence.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return result

    validation.SFTTrainingEngine.optimizer_step = audited_step
    if args.stop_after:
        original_continue = validation.SFTTrainer.should_continue_training

        def bounded_training(trainer):
            return (
                original_continue(trainer)
                and trainer.engine.num_completed_steps < args.stop_after
            )

        validation.SFTTrainer.should_continue_training = bounded_training
    entry.main(training_args)


if __name__ == "__main__":
    main()
