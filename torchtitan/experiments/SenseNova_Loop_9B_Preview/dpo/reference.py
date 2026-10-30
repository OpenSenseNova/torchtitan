# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Independent sharded reference runtime without an optimizer or new meshes."""

import copy
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.distributed.tensor import DTensor


@contextmanager
def preserve_rng():
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda)


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def physical_state(model):
    """Canonical physical parameters/buffers without logical save-hook gathers."""
    from torchtitan.components.checkpointer.utils import canonical_fqn
    from torch.distributed.fsdp import FSDPModule

    # FSDP's root may keep BF16 all-gather views after an eval/no-grad forward.
    # Persist the master shards, not those temporary compute parameters.
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()

    state = {canonical_fqn(name): value for name, value in model.named_parameters()}
    for prefix, module in model.named_modules():
        for name, value in module.named_buffers(recurse=False):
            if name not in module._non_persistent_buffers_set:
                key = f"{prefix}.{name}" if prefix else name
                state[canonical_fqn(key)] = value
    return state


def copy_local_state(source, destination):
    with torch.no_grad():
        a, b = physical_state(source), physical_state(destination)
        if a.keys() != b.keys():
            raise ValueError("Policy/reference state keys differ")
        for name in a:
            src, dst = local_tensor(a[name]), local_tensor(b[name])
            if src.shape != dst.shape or src.dtype != dst.dtype:
                raise ValueError(f"Policy/reference shard mismatch: {name}")
            if src.numel() and src.data_ptr() == dst.data_ptr():
                raise ValueError(f"Policy/reference share mutable storage: {name}")
            dst.copy_(src)


class ReferenceRuntime:
    def __init__(self, engine, *, copy_policy=True):
        from torchtitan.config import TORCH_DTYPE_MAP
        from torchtitan.tools.utils import set_default_dtype

        config = copy.deepcopy(engine.model_config)
        config.trainable_loop_only = False
        self.parts = []
        with preserve_rng(), engine.train_context():
            with (
                torch.device("meta"),
                set_default_dtype(TORCH_DTYPE_MAP[engine.config.training.dtype]),
            ):
                model = config.build()
            model.requires_grad_(False)
            model = engine.model_spec.parallelize_fn(
                model,
                parallel_dims=engine.parallel_dims,
                training=engine.config.training,
                parallelism=engine.config.parallelism,
                compile_config=None,
                ac_config=None,
                dump_folder=engine.output_dir,
            )
            model.to_empty(device=engine.device)
            with torch.no_grad():
                model.init_weights(buffer_device=None)
            model.requires_grad_(False)
            model.eval()
            model._skip_lm_head = True
            if copy_policy:
                copy_local_state(engine.model_parts[0], model)
            self.parts = [model]
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Reference contains trainable parameters")

    @property
    def model(self):
        return self.parts[0]

    def reshard(self):
        from torch.distributed.fsdp import FSDPModule

        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
