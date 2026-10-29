# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded CPU-only final-pack preparation; consumption owns the data cursor."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass
from threading import Condition

import torch


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Pack prefetch must produce CPU tensors only")
        return value.numel() * value.element_size()
    if is_dataclass(value) and not isinstance(value, type):
        return sum(tensor_bytes(getattr(value, f.name)) for f in fields(value))
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class PackPrefetch:
    """One producer, at most `depth` futures, and a byte bound on ready packs.

    One oversized pack is allowed when the ready queue is empty. The producer
    can additionally hold one in-flight pack and its temporary encoding buffers;
    the byte bound is not a bound on total process RSS or consumer-held batches.
    Neither the producer nor build() may issue CUDA/distributed operations.
    """

    def __init__(self, build, *, depth, max_bytes):
        if depth < 1 or max_bytes < 1:
            raise ValueError("Pack prefetch depth and byte limit must be positive")
        self.build, self.depth, self.max_bytes = build, depth, max_bytes
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arrow-pack")
        self.condition = Condition()
        self.futures = {}
        self.ready_bytes = self.peak_ready_bytes = 0
        self.closed = False

    def _produce(self, indices):
        result = self.build(indices)
        size = tensor_bytes(result[0])
        with self.condition:
            while (
                not self.closed
                and self.ready_bytes
                and self.ready_bytes + size > self.max_bytes
            ):
                self.condition.wait()
            if self.closed:
                raise RuntimeError("Pack prefetch closed")
            self.ready_bytes += size
            self.peak_ready_bytes = max(self.peak_ready_bytes, self.ready_bytes)
        return result, size

    def schedule(self, entries):
        entries = list(entries)
        if len(entries) > self.depth or len({key for key, _ in entries}) != len(
            entries
        ):
            raise ValueError("Invalid pack prefetch window")
        wanted = {key for key, _ in entries}
        if self.futures.keys() - wanted:
            raise RuntimeError(
                "Unconsumed pack would be discarded; close before restoring a cursor"
            )
        if self.closed:
            raise RuntimeError("Pack prefetch closed")
        for key, indices in entries:
            if key not in self.futures:
                self.futures[key] = self.pool.submit(self._produce, tuple(indices))

    def get(self, key):
        # Retain a failing future until close, so retries cannot reorder work.
        result, size = self.futures[key].result()
        del self.futures[key]
        with self.condition:
            self.ready_bytes -= size
            self.condition.notify_all()
        return result

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.futures.clear()
        self.ready_bytes = 0
