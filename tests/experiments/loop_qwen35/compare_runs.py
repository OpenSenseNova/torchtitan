# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Compare exact losses and sampled gradients across every distributed rank."""

import argparse
import json
import math
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-step", type=int, default=1)
    args = parser.parse_args()
    differences, counts = [], {}
    rank_files = sorted(args.left.glob("numerics-rank-*.jsonl"))
    assert rank_files, f"No numerical evidence in {args.left}"
    assert {p.name for p in rank_files} == {
        p.name for p in args.right.glob("numerics-rank-*.jsonl")
    }, "Distributed rank sets differ"
    for rank_file in rank_files:
        rank = int(rank_file.stem.rsplit("-", 1)[1])
        left = records(args.left / f"numerics-rank-{rank}.jsonl")
        right = records(args.right / f"numerics-rank-{rank}.jsonl")
        # Optimizer evidence is captured immediately after the update.
        left = {row["step"]: row for row in left if row["step"] >= args.start_step}
        right = {row["step"]: row for row in right if row["step"] >= args.start_step}
        assert left and left.keys() == right.keys(), (rank, left.keys(), right.keys())
        counts[str(rank)] = len(left)
        for step, row in left.items():
            if row != right[step]:
                differences.append(
                    dict(
                        rank=rank,
                        completed_steps=step,
                        gradient_mismatches=[
                            key
                            for key, value in row["gradient_samples"].items()
                            if right[step]["gradient_samples"].get(key) != value
                        ],
                        grad_norms=[row["grad_norm"], right[step]["grad_norm"]],
                    )
                )

    def metrics(directory):
        found = {}
        for file in (directory / "logs").glob("experiment_metrics_*.jsonl"):
            for row in records(file):
                if row.get("event") == "train" and row["step"] >= args.start_step:
                    found[row["step"]] = row["metrics"]
        return found

    left_metrics, right_metrics = metrics(args.left), metrics(args.right)
    assert left_metrics and left_metrics.keys() == right_metrics.keys()
    losses = []
    for step, row in left_metrics.items():
        other = right_metrics[step]
        for key in (
            "loss_metrics/global_avg_loss",
            "grad_norm",
            "loss_metrics/global_max_loss",
            "loss/train",
        ):
            if key in row:
                assert math.isfinite(row[key]) and math.isfinite(other[key])
                if row[key] != other[key]:
                    differences.append(
                        dict(step=step, metric=key, left=row[key], right=other[key])
                    )
        losses.append(
            dict(
                step=step,
                loss=row.get("loss_metrics/global_avg_loss", row.get("loss/train")),
                other_loss=other.get(
                    "loss_metrics/global_avg_loss", other.get("loss/train")
                ),
            )
        )
    report = dict(
        exact=not differences, rank_steps=counts, losses=losses, differences=differences
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if differences:
        raise AssertionError("Numerical comparison failed")


if __name__ == "__main__":
    main()
