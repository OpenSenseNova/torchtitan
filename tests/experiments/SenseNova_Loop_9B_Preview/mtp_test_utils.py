# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Small independent assertions for MTP numerics and frozen parameters."""

import hashlib
import torch
from torch.distributed.tensor import DTensor


def local_cpu(value):
    if isinstance(value, DTensor):
        value = value.to_local()
    return value.detach().cpu().contiguous()


def tensor_manifest(value):
    value = local_cpu(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(
            value.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "finite": bool(torch.isfinite(value).all()),
    }


def compare_tensor(actual, reference, *, kind, name):
    actual, reference = local_cpu(actual), local_cpu(reference)
    assert kind == "exact"
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, msg=name)
    return {}


class FrozenAudit:
    """Check every local shard, gradient and optimizer state of frozen weights."""

    def __init__(self, model):
        self.before = {}
        for name, parameter in model.named_parameters():
            if name.startswith("mtp_layers."):
                assert parameter.requires_grad, f"MTP unexpectedly frozen: {name}"
            else:
                assert (
                    not parameter.requires_grad
                ), f"Backbone unexpectedly trainable: {name}"
                self.before[name] = local_cpu(parameter).clone()
        assert self.before, "No frozen parameters to audit"

    def check(self, model, optimizers):
        parameters = dict(model.named_parameters())
        actual_frozen = {
            name for name in parameters if not name.startswith("mtp_layers.")
        }
        assert actual_frozen == self.before.keys(), "Frozen parameter set changed"
        for name, saved in self.before.items():
            parameter = parameters[name]
            assert not parameter.requires_grad and parameter.grad is None, name
            compare_tensor(parameter, saved, kind="exact", name=name)
            for optimizer in optimizers:
                assert (
                    parameter not in optimizer.state
                ), f"Frozen optimizer state: {name}"
                assert all(
                    parameter is not p
                    for group in optimizer.param_groups
                    for p in group["params"]
                ), name
        return {
            "frozen_parameters": len(self.before),
            "values_exact": True,
            "gradients_none": True,
            "optimizer_membership": False,
            "optimizer_state": False,
        }


def logical_mtp_gradients(block):
    """Split physical FFN gradients just as logical checkpoint tensors split."""
    gradients = {}
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"Missing MTP gradient: {name}"
        gradient = parameter.grad.detach()
        assert torch.isfinite(gradient).all(), f"Non-finite MTP gradient: {name}"
        if name == "feed_forward.w13.weight":
            gate_up = gradient.unflatten(0, (-1, 2))
            gradients["feed_forward.w1.weight"] = gate_up[:, 0].contiguous()
            gradients["feed_forward.w3.weight"] = gate_up[:, 1].contiguous()
        else:
            gradients[name] = gradient
    return gradients


def state_fingerprint(value):
    import numpy as np

    if isinstance(value, torch.Tensor):
        return tensor_manifest(value)
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): state_fingerprint(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [state_fingerprint(item) for item in value]
    return value
