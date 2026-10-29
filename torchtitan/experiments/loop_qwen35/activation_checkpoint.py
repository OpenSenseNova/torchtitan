# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint Loop blocks while optionally retaining FA2 attention outputs."""

from dataclasses import dataclass

from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC

from torchtitan.models.common.attention_compat import varlen_attention_op


class AttentionSelectiveAC(SelectiveAC):
    """Keep FA2 outputs; recompute projections, routers, MLPs and CP exchanges."""

    @dataclass(kw_only=True, slots=True)
    class Config(SelectiveAC.Config):
        pass

    def get_save_ops(self):
        # The custom op hides its aten backend from selective checkpoint modes.
        return {varlen_attention_op()}

    def _wrap_block(self, module, *, base_fqn=None):
        # The same policy also wraps vision blocks. Retain only decoder FA2.
        if not getattr(module, "full_attn", False):
            full = FullAC(
                FullAC.Config(
                    preserve_rng_state=self.config.preserve_rng_state,
                    determinism_check=self.config.determinism_check,
                    debug=self.config.debug,
                )
            )
            return full._wrap_block(module, base_fqn=base_fqn)
        return super()._wrap_block(module, base_fqn=base_fqn)


def checkpoint_config(enabled, policy):
    if policy not in ("full", "attention"):
        raise ValueError("Checkpoint policy must be full or attention")
    if not enabled:
        return None
    return AttentionSelectiveAC.Config() if policy == "attention" else FullAC.Config()
