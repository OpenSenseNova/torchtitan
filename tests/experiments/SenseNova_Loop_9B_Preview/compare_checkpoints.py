# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Compare every saved model and optimizer tensor on every distributed rank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-data", action="store_true")
    parser.add_argument("--compare-cpu-rng", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    left = torch.load(
        args.left / f"rank-{rank}.pt", map_location="cpu", weights_only=False, mmap=True
    )
    right = torch.load(
        args.right / f"rank-{rank}.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    failures, counts = [], dict(tensors=0, elements=0)

    def compare(a, b, path):
        if isinstance(a, DTensor):
            if (
                not isinstance(b, DTensor)
                or a.placements != b.placements
                or a.shape != b.shape
            ):
                failures.append(path + ": DTensor layout differs")
                return
            a, b = a.to_local(), b.to_local()
        if isinstance(a, torch.Tensor):
            counts["tensors"] += 1
            counts["elements"] += a.numel()
            if (
                not isinstance(b, torch.Tensor)
                or a.dtype != b.dtype
                or not torch.equal(a, b)
            ):
                failures.append(path + ": tensor differs")
        elif isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                failures.append(path + ": array differs")
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                failures.append(path + ": keys differ")
            for key in a.keys() & b.keys():
                compare(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                failures.append(path + ": lengths differ")
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}/{index}")
        elif a != b:
            failures.append(path + ": value differs")

    keys = ["model", "optimizer", "scheduler", "train_state", "torch_rng", "cuda_rng"]
    # Independent runs do not seed the unused global Python/NumPy RNGs.
    # A resumed run must restore those states too, so compare them explicitly.
    if args.compare_cpu_rng:
        keys += ["python_rng", "numpy_rng"]
    if args.compare_data:
        keys.append("dataloader")
    for key in keys:
        compare(left[key], right[key], key)
    report = dict(rank=rank, exact=not failures, **counts, failures=failures)
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report)
    if rank == 0:
        output = dict(
            exact=all(row["exact"] for row in reports), keys=keys, ranks=reports
        )
        args.output.write_text(json.dumps(output, indent=2))
        print(json.dumps(output), flush=True)
    dist.destroy_process_group()
    if any(not row["exact"] for row in reports):
        raise AssertionError("Checkpoint comparison failed")


if __name__ == "__main__":
    main()
