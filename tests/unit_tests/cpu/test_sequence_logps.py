# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the BSD-style license in the root LICENSE file.
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp import FSDPModule

from torchtitan.components.sequence_logps import sequence_logps, weighted_sequence_backward


class RecordingHead(FSDPModule, torch.nn.Linear):
    def __new__(cls):
        return object.__new__(cls)

    def __init__(self):
        super().__init__(4, 8, bias=False)
        self.group = SimpleNamespace(reduce_grads=True, all_reduce_grads=False,
                                     _reshard_after_forward=True, reshard_after_backward=False)
        self.states = []
        self.calls = 0
        self.fail_at = -1
        self.weight.register_post_accumulate_grad_hook(
            lambda parameter: self.states.append((self.group.reduce_grads, self.group.all_reduce_grads)))

    def _get_fsdp_state(self):
        return SimpleNamespace(_fsdp_param_group=self.group)

    def set_reshard_after_forward(self, value):
        self.group._reshard_after_forward = value

    def set_reshard_after_backward(self, value):
        self.group.reshard_after_backward = value

    def unshard(self):
        pass

    def reshard(self):
        pass

    def forward(self, value):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError('injected head failure')
        return super().forward(value)


@pytest.mark.parametrize('all_reduce', [True, False])
@pytest.mark.parametrize('fail', [True, False])
def test_sync_and_reshard_flags_restored(all_reduce, fail):
    head = RecordingHead()
    head.set_requires_all_reduce(all_reduce, recurse=False)
    head.fail_at = 2 if fail else -1
    hidden = torch.randn(8, 4, requires_grad=True)
    labels = torch.arange(8)
    ids = torch.arange(8) // 4
    if fail:
        with pytest.raises(ValueError, match='injected'):
            weighted_sequence_backward(hidden, head, labels, ids, torch.tensor([1., -1.]), num_chunks=4)
    else:
        weighted_sequence_backward(hidden, head, labels, ids, torch.tensor([1., -1.]), num_chunks=4)
        assert head.states == [(False, False)] * 3 + [(True, all_reduce)]
    assert (head.group.reduce_grads, head.group.all_reduce_grads) == (True, all_reduce)
    assert head.group._reshard_after_forward and not head.group.reshard_after_backward


def test_reference_scoring_has_no_gradient_and_restores_flags():
    head = RecordingHead()
    with torch.no_grad():
        scores = sequence_logps(torch.randn(8, 4), head, torch.arange(8), torch.arange(8) // 4, 2, num_chunks=4)
    assert not scores.requires_grad and head.weight.grad is None
    assert head.group.reduce_grads and not head.group.all_reduce_grads
    assert head.group._reshard_after_forward and not head.group.reshard_after_backward
