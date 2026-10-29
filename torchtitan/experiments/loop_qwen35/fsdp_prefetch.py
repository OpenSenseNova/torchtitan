# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit FSDP prefetch for repeated decoder blocks with nested mHC routers."""


def loop_prefetch_neighbors(num_layers, start, end, repeated):
    """Return possible successors and predecessors in execution order."""
    if not 0 <= start < end <= num_layers:
        raise ValueError("Invalid Loop interval for FSDP prefetch")
    forward = [[i + 1] if i + 1 < num_layers else [] for i in range(num_layers)]
    backward = [[i - 1] if i else [] for i in range(num_layers)]
    if repeated and end - start > 1:
        forward[end - 1].insert(0, start)
        backward[start].insert(0, end - 1)
    return forward, backward


def configure_loop_fsdp_prefetch(model):
    """Prefetch one decoder block and its separately sharded FP32 routers.

    At a Loop boundary both possible neighbors are prefetched. This supports
    changing repeat counts without changing collective order across ranks,
    at the cost of retaining extra boundary parameters until they are used. FSDP owns
    the communication streams, waits, parameter lifetime and AC recomputation.
    """
    from torch.distributed.fsdp import FSDPModule

    layers = list(model.layers.values())
    settings = model.loop_settings
    repeated = max(*settings.counts, settings.eval_count) > 1
    forward, backward = loop_prefetch_neighbors(
        len(layers), settings.start, settings.end, repeated
    )

    def routers(layer):
        return [
            getattr(layer, name)
            for name in ("mhc_attn", "mhc_ffn")
            if isinstance(getattr(layer, name, None), FSDPModule)
        ]

    def targets(layer, neighbors):
        # Current routers must precede a large next-block all-gather in the
        # communication queue: they are consumed immediately in this block.
        result = routers(layer)
        for index in neighbors:
            target = layers[index]
            result.extend([target, *routers(target)])
        return result

    if not all(isinstance(layer, FSDPModule) for layer in layers):
        raise ValueError("Loop prefetch must be configured after decoder FSDP")
    for index, layer in enumerate(layers):
        layer.set_modules_to_forward_prefetch(targets(layer, forward[index]))
        layer.set_modules_to_backward_prefetch(targets(layer, backward[index]))
