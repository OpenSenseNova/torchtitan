# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Process-local kernel selection for the Loop training recipe."""

import importlib
import logging

import torch
import triton


class _StableGDNOutput:
    def __init__(self, kernel):
        # Keep the upstream varlen heuristic and JIT function, bypass autotuning.
        self.kernel = triton.heuristics(kernel.values)(kernel.fn.fn)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.update(BK=32, BV=32, num_warps=2, num_stages=3)
            return self.kernel[grid](*args, **kwargs)

        return launch


def stabilize_gdn_output() -> None:
    """Select the validated output tile without modifying installed packages.

    On SM103 with attn-gym 0.0.8 / Triton 3.6, the autotuned 128x128, 8-warp
    MMA5 path can produce different outputs for byte-identical inputs. The
    32x32, 2-warp path avoids that specialization. Keep this selection explicit
    until a replacement toolchain passes forward/backward replay validation.
    MACA FLA exposes the same tile contract; select it without timing variance.
    """
    module_name = (
        "fla.ops.common.chunk_o"
        if getattr(torch.version, "maca", None)
        else "attn_gym.linear.gdn.fwd.triton.chunk_gdn_fwd_output"
    )
    module = importlib.import_module(module_name)
    kernel = module.chunk_fwd_kernel_o
    if isinstance(kernel, _StableGDNOutput):
        return
    if (
        not isinstance(kernel, triton.runtime.Heuristics)
        or not isinstance(kernel.fn, triton.runtime.Autotuner)
        or not any(
            c.kwargs == {"BK": 32, "BV": 32} and c.num_warps == 2
            for c in kernel.fn.configs
        )
    ):
        raise RuntimeError(
            "Unsupported GDN output API; validate this environment "
            "before selecting --gdn-output-kernel native"
        )
    module.chunk_fwd_kernel_o = _StableGDNOutput(kernel)
    logging.getLogger(__name__).info("GDN output: stable 32x32 tile, 2 warps, 3 stages")
