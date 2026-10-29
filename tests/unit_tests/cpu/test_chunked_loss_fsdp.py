# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torchtitan.components.loss import ChunkedLossWrapper


class _RecordingFSDPLinear(FSDPModule, nn.Linear):
    """Record backward flags using real FSDP setters without communication."""

    def __new__(cls, *args, **kwargs):
        # Keep the FSDP mixin so the real synchronization setters are exercised.
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fsdp_state = SimpleNamespace(
            _fsdp_param_group=SimpleNamespace(reduce_grads=True, all_reduce_grads=True)
        )
        self.backward_sync_states: list[tuple[bool, bool]] = []
        self.weight.register_post_accumulate_grad_hook(self._record_backward_sync)

    def _get_fsdp_state(self):
        return self.fsdp_state

    def _record_backward_sync(self, param):
        group = self.fsdp_state._fsdp_param_group
        self.backward_sync_states.append((group.reduce_grads, group.all_reduce_grads))

    def set_reshard_after_forward(self, enabled):
        pass

    def set_reshard_after_backward(self, enabled):
        pass

    def unshard(self):
        pass

    def reshard(self):
        pass


class TestChunkedLossFSDPSync(unittest.TestCase):
    def test_fsdp_preserves_outer_sync_state(self):
        torch.manual_seed(42)
        for num_chunks in (1, 4):
            for reduce_grads, all_reduce_grads in (
                (True, False),
                (True, True),
                (False, False),
                (False, True),
            ):
                for requires_grad in (True, False):
                    with self.subTest(
                        num_chunks=num_chunks,
                        reduce_grads=reduce_grads,
                        all_reduce_grads=all_reduce_grads,
                        requires_grad=requires_grad,
                    ):
                        head = _RecordingFSDPLinear(4, 8, bias=False)
                        head.set_requires_gradient_sync(reduce_grads, recurse=False)
                        head.set_requires_all_reduce(all_reduce_grads, recurse=False)
                        chunked_loss = ChunkedLossWrapper(
                            ChunkedLossWrapper.Config(num_chunks=num_chunks)
                        )
                        chunked_loss.set_lm_head(head)
                        hidden = torch.randn(8, 4, requires_grad=requires_grad)
                        labels = torch.randint(0, 8, (8,))

                        loss, _ = chunked_loss(hidden, labels)
                        if requires_grad:
                            loss.backward()

                        expected_state = (reduce_grads, all_reduce_grads)
                        self.assertEqual(
                            head.backward_sync_states,
                            [(False, False)] * (num_chunks - 1) + [expected_state]
                            if requires_grad
                            else [],
                        )
                        group = head._get_fsdp_state()._fsdp_param_group
                        self.assertEqual(
                            (group.reduce_grads, group.all_reduce_grads), expected_state
                        )

    def test_fsdp_accumulation_preserves_loss_and_gradients(self):
        torch.manual_seed(42)
        num_chunks = 4
        head = _RecordingFSDPLinear(4, 8, bias=False)
        reference_head = nn.Linear(4, 8, bias=False)
        reference_head.load_state_dict(head.state_dict())
        chunked_loss = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(num_chunks=num_chunks)
        )
        reference_loss = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(num_chunks=num_chunks)
        )
        chunked_loss.set_lm_head(head)
        reference_loss.set_lm_head(reference_head)
        all_reduce_schedule = (False, False, True)
        global_valid_tokens = torch.tensor(8 * len(all_reduce_schedule))

        for requires_all_reduce in all_reduce_schedule:
            head.set_requires_all_reduce(requires_all_reduce)
            hidden = torch.randn(8, 4, requires_grad=True)
            reference_hidden = hidden.detach().clone().requires_grad_(True)
            labels = torch.randint(0, 8, (8,))
            loss, _ = chunked_loss(hidden, labels, global_valid_tokens)
            expected_loss, _ = reference_loss(
                reference_hidden, labels, global_valid_tokens
            )
            loss.backward()
            expected_loss.backward()

            torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
            torch.testing.assert_close(
                hidden.grad, reference_hidden.grad, rtol=0, atol=0
            )
            torch.testing.assert_close(
                head.weight.grad, reference_head.weight.grad, rtol=0, atol=0
            )

        expected_sync_states = []
        for requires_all_reduce in all_reduce_schedule:
            expected_sync_states.extend(
                [(False, False)] * (num_chunks - 1) + [(True, requires_all_reduce)]
            )
        self.assertEqual(head.backward_sync_states, expected_sync_states)
