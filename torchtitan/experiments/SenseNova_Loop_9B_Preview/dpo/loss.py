# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Sigmoid DPO over complete response scores (pairs, chosen/rejected)."""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


def complete_sequence_scores(local_scores, group=None):
    """CP forward SUM, backward identity; parameter FSDP performs gradient SUM."""
    if group is None:
        return local_scores
    complete = local_scores.detach().clone()
    dist.all_reduce(complete, group=group)
    return complete + (local_scores - local_scores.detach())


def dpo_loss(policy, reference, valid, *, beta=0.1, global_pairs=None):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    if policy.ndim != 2 or policy.shape[1] != 2 or reference.shape != policy.shape:
        raise ValueError("DPO scores must have shape [pairs, 2]")
    if valid.shape != policy.shape[:1] or valid.dtype != torch.bool:
        raise ValueError(
            "DPO validity must be a boolean vector with one element per pair"
        )
    denominator = valid.sum() if global_pairs is None else global_pairs
    if denominator <= 0:
        raise ValueError("DPO requires at least one globally valid pair")
    rewards = beta * (policy - reference.detach())
    margin = rewards[:, 0] - rewards[:, 1]
    per_pair = -F.logsigmoid(margin)
    loss = torch.where(valid, per_pair, 0).sum() / denominator
    # Fixed coefficients for the second lm_head pass. No higher order graph.
    chosen = -beta * torch.sigmoid(-margin.detach()) * valid / denominator
    coefficients = torch.stack((chosen, -chosen), dim=-1)

    def total(value):
        return torch.where(valid, value.detach(), 0).sum()

    metrics = {
        "loss": total(per_pair),
        "policy_chosen_logps": total(policy[:, 0]),
        "policy_rejected_logps": total(policy[:, 1]),
        "reference_chosen_logps": total(reference[:, 0]),
        "reference_rejected_logps": total(reference[:, 1]),
        "chosen_reward": total(rewards[:, 0]),
        "rejected_reward": total(rewards[:, 1]),
        "reward_margin": total(margin),
        "preference_accuracy": total((margin > 0).to(policy.dtype)),
        "valid_pairs": valid.sum().detach(),
    }
    return loss, coefficients, metrics
