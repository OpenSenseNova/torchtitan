# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the BSD-style license in the root LICENSE file.
"""Complete sequence log probabilities and bounded-memory weighted gradients.

T: local tokens, D: hidden width, V: vocabulary, S: response sequences.
The caller owns nonlinear objectives and cross-rank score reductions.
"""
from contextlib import contextmanager

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed._composable.fsdp import FSDPModule

from torchtitan.components.loss import compute_logprobs


@contextmanager
def unsharded_head(head):
    """Keep a head resident and coalesce gradients, restoring caller flags."""
    group = head._get_fsdp_state()._fsdp_param_group if isinstance(head, FSDPModule) else None
    if group is None:
        yield lambda: None
        return
    sync, all_reduce = group.reduce_grads, group.all_reduce_grads
    reshard_forward, reshard_backward = group._reshard_after_forward, group.reshard_after_backward

    def restore_sync():
        head.set_requires_gradient_sync(sync, recurse=False)
        head.set_requires_all_reduce(all_reduce, recurse=False)

    try:
        head.set_reshard_after_forward(False)
        head.set_reshard_after_backward(False)
        head.set_requires_gradient_sync(False, recurse=False)
        with spmd.no_typecheck():
            head.unshard()
        yield restore_sync
    finally:
        restore_sync()
        head.set_reshard_after_forward(reshard_forward)
        head.set_reshard_after_backward(reshard_backward)
        head.reshard()


def token_logps(logits_TV, labels_T):
    # Keep FP64 for numerical reference tests; production accumulates in FP32.
    if logits_TV.dtype == torch.float64:
        return -F.cross_entropy(logits_TV, labels_T, reduction='none', ignore_index=-100)
    return compute_logprobs(logits_TV, labels_T)


def segment_logps(logits_TV, labels_T, sequence_ids_T, num_sequences):
    values_T = token_logps(logits_TV, labels_T)
    valid_T = (labels_T != -100) & (sequence_ids_T >= 0)
    return values_T.new_zeros(num_sequences).scatter_add(
        0, sequence_ids_T.clamp_min(0), torch.where(valid_T, values_T, 0))


def _slices(length, num_chunks):
    if num_chunks < 1 or length % num_chunks:
        raise ValueError('Sequence length must be divisible by the positive chunk count')
    size = length // num_chunks
    return [slice(start, start + size) for start in range(0, length, size)]


def sequence_logps(hidden_TD, head, labels_T, sequence_ids_T, num_sequences, *, num_chunks=0):
    """Full autograd baseline, or a no-grad chunked score pass."""
    if num_chunks == 0:
        with spmd.local():
            return segment_logps(head(hidden_TD), labels_T, sequence_ids_T, num_sequences)
    if torch.is_grad_enabled() and hidden_TD.requires_grad:
        raise ValueError('Chunked scoring is no-grad; use weighted_sequence_backward for gradients')
    with spmd.local(), unsharded_head(head):
        scores_S = None
        for section in _slices(hidden_TD.shape[0], num_chunks):
            partial_S = segment_logps(head(hidden_TD[section]), labels_T[section], sequence_ids_T[section],
                                     num_sequences)
            scores_S = partial_S if scores_S is None else scores_S + partial_S
        return scores_S


def weighted_sequence_backward(hidden_TD, head, labels_T, sequence_ids_T, coefficients_S, *, num_chunks):
    """Backpropagate fixed response-score derivatives with one decoder backward.

Each head graph is consumed immediately. This also accumulates gradients when
the head weight is shared with an embedding used by the original decoder.
"""
    with spmd.local(), unsharded_head(head) as synchronize:
        sections = _slices(hidden_TD.shape[0], num_chunks)
        gradient_TD = torch.zeros_like(hidden_TD, dtype=(
            torch.float64 if hidden_TD.dtype == torch.float64 else torch.float32))
        for index, section in enumerate(sections):
            if index == len(sections) - 1:
                synchronize()
            leaf_TD = hidden_TD[section].detach().requires_grad_(True)
            scores_S = segment_logps(head(leaf_TD), labels_T[section], sequence_ids_T[section],
                                     coefficients_S.numel())
            objective = (scores_S * coefficients_S.detach()).sum()
            with spmd.no_typecheck():
                objective.backward()
            gradient_TD[section] = leaf_TD.grad
    with spmd.no_typecheck():
        hidden_TD.backward(gradient_TD.to(hidden_TD.dtype))
