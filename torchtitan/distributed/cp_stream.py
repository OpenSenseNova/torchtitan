# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schedule native CP redistributions without waiting on the compute stream."""

from dataclasses import dataclass

import spmd_types as spmd
import torch

_streams: dict[int, torch.cuda.Stream] = {}


@dataclass
class PendingCPRedistribute:
    tensor: torch.Tensor
    ready: torch.cuda.Event

    def wait(self) -> torch.Tensor:
        stream = torch.cuda.current_stream(self.tensor.device)
        stream.wait_event(self.ready)
        self.tensor.record_stream(stream)
        return self.tensor


def redistribute_on_cp_stream(
    tensor: torch.Tensor, group, *, src, dst
) -> PendingCPRedistribute:
    """Preserve native SPMD/autograd rules and wait only at the first consumer.

    One stream per device bounds allocator pools when many decoder blocks reuse
    this operation. Input and output storage are recorded on their consumers.
    PyTorch autograd owns the corresponding backward stream dependencies.
    """
    if tensor.device.type != "cuda":
        raise ValueError("CP stream overlap requires CUDA tensors")
    device = tensor.device.index
    stream = _streams.get(device)
    if stream is None:
        stream = torch.cuda.Stream(device=tensor.device)
        _streams[device] = stream
    stream.wait_stream(torch.cuda.current_stream(tensor.device))
    with torch.cuda.stream(stream):
        output = spmd.redistribute(tensor, group, src=src, dst=dst)
        ready = torch.cuda.Event()
        ready.record(stream)
    tensor.record_stream(stream)
    return PendingCPRedistribute(output, ready)
