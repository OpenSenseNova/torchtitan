# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Pair-weighted metrics; CP replicas are deliberately excluded."""

import torch
import torch.distributed as dist


def reduce_metrics(totals, parallel_dims):
    keys = list(totals)
    values = torch.stack([totals[key].double() for key in keys])
    mesh = parallel_dims.get_optional_mesh("dp")
    if mesh is not None:
        dist.all_reduce(values, group=mesh.get_group())
    result = dict(zip(keys, values.tolist()))
    pairs = result["valid_pairs"]
    if pairs <= 0:
        raise ValueError("DPO step has no globally valid pairs")
    for key in result:
        if key not in ("valid_pairs", "chosen_tokens", "rejected_tokens"):
            result[key] /= pairs
    return result


def add_metrics(totals, current):
    for key, value in current.items():
        totals[key] = totals.get(key, 0) + value


def synchronized_error(error, device):
    """Complete data validation on every rank before any model collective."""
    failed = torch.tensor(int(error is not None), device=device)
    if dist.is_initialized():
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        messages = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
        if dist.is_initialized():
            dist.all_gather_object(messages, str(error) if error is not None else None)
        else:
            messages[0] = str(error)
        raise ValueError(
            "DPO coordinated failure: " + "; ".join(sorted({m for m in messages if m}))
        ) from error
