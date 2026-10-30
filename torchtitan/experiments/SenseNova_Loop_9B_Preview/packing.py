# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic whole-row packing, grouped by attention cost across DP ranks."""

import numpy as np

PLAN_VERSION = 5


def read_lengths(column):
    """Sum Arrow list lengths without materializing a million Python records."""
    import pyarrow as pa

    result = []
    for chunk in column.chunks:
        if chunk.null_count:
            raise ValueError("Balanced packing requires non-null lengths metadata")
        if pa.types.is_list(chunk.type) or pa.types.is_large_list(chunk.type):
            values = chunk.values
            if values.null_count or not pa.types.is_integer(values.type):
                raise ValueError("Balanced packing requires integer lengths metadata")
            offsets = chunk.offsets.to_numpy()
            if (
                values.slice(int(offsets[0]), int(offsets[-1] - offsets[0])).to_numpy()
                < 0
            ).any():
                raise ValueError(
                    "Balanced packing requires nonnegative length components"
                )
            prefix = np.concatenate(([0], np.cumsum(values.to_numpy(), dtype=np.int64)))
            lengths = prefix[offsets[1:]] - prefix[offsets[:-1]]
        elif pa.types.is_integer(chunk.type):
            lengths = chunk.to_numpy().astype(np.int64)
        else:
            raise ValueError(
                "Balanced packing requires integer or list-of-integer lengths"
            )
        if (lengths < 0).any():
            raise ValueError(
                "Balanced packing requires nonnegative row lengths (zero means excluded offline)"
            )
        result.append(lengths)
    return np.concatenate(result)


def attention_cost(lengths, token_budget, max_length):
    """Include the padded tail as a separate attention segment."""
    return sum(n * n for n in lengths) + (token_budget - sum(lengths)) ** 2


def balanced_plan(
    lengths,
    *,
    token_budget,
    max_length,
    dp_world_size,
    seed,
    epoch,
    buffer_size,
    vision_costs=None,
    grad_accumulation=1,
    dp_balance=True,
):
    """Plan exact lengths once; never repair or change owners during training.

    Zero marks an offline-excluded row. Positive over-limit lengths are errors,
    not clipped estimates. Rebalance complete documents at the epoch tail before
    adding unavoidable zero-loss slots. Every positive row occurs exactly once.
    Disabling DP balance shuffles whole packs without cost grouping or GA
    smoothing; bin membership, strict lengths and tail handling stay unchanged.
    """
    if min(token_budget, max_length, dp_world_size, buffer_size) < 1:
        raise ValueError("Packing sizes must be positive")
    if buffer_size < dp_world_size:
        raise ValueError("Packing buffer must contain at least one row per DP rank")
    lengths = np.asarray(lengths)
    if (
        lengths.ndim != 1
        or not np.issubdtype(lengths.dtype, np.integer)
        or (lengths < 0).any()
    ):
        raise ValueError("Exact packing lengths must be nonnegative integers")
    if grad_accumulation < 1:
        raise ValueError("grad_accumulation must be positive")
    if vision_costs is not None:
        vision_costs = np.asarray(vision_costs, dtype=np.float64)
        if (
            vision_costs.shape != lengths.shape
            or not np.isfinite(vision_costs).all()
            or (vision_costs < 0).any()
        ):
            raise ValueError(
                "Visual packing costs must be finite nonnegative values for every row"
            )
    limit = min(max_length, token_budget)
    invalid = np.flatnonzero(lengths > limit)
    if len(invalid):
        index = int(invalid[0])
        raise ValueError(
            f"Row {index} has offline length {lengths[index]} > packing limit {limit}; "
            "filter or split it explicitly during preprocessing"
        )

    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(np.flatnonzero(lengths > 0))
    if not len(order):
        raise ValueError("No usable SFT sample in the exact packing lengths")

    def cost(ids):
        decoder = attention_cost(
            [int(lengths[i]) for i in ids], token_budget, max_length
        )
        return (
            decoder
            if vision_costs is None
            else decoder + sum(vision_costs[i] for i in ids)
        )

    plan, carry = [], []
    for start in range(0, len(order), buffer_size):
        window = [(int(i), int(lengths[i])) for i in order[start : start + buffer_size]]
        bins, _ = calculate_matched_group(window, token_budget, is_finished=True)
        candidates = carry + [tuple(item[0] for item in pack) for pack in bins]
        # Preserve v5 pack membership, including carry and tail redistribution.
        # Only scheduling changes; image costs must not silently repack rows.
        candidates.sort(
            key=lambda ids: attention_cost(
                [int(lengths[i]) for i in ids], token_budget, max_length
            )
        )
        count = len(candidates) // dp_world_size
        carry = candidates[count * dp_world_size :]
        candidates = candidates[: count * dp_world_size]
        if not dp_balance:
            candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
        elif vision_costs is not None:
            candidates.sort(key=cost)
        groups = [
            candidates[i * dp_world_size : (i + 1) * dp_world_size]
            for i in range(count)
        ]
        for group_index in rng.permutation(count):
            group = groups[int(group_index)]
            plan.extend(group[int(i)] for i in rng.permutation(dp_world_size))
    if dp_balance and vision_costs is not None and grad_accumulation > 1:
        # Smooth complete accumulation windows across the entire epoch, not
        # independently per packing buffer (which can misalign step boundaries).
        groups = [
            plan[i : i + dp_world_size] for i in range(0, len(plan), dp_world_size)
        ]
        groups.sort(key=lambda group: max(cost(pack) for pack in group))
        steps = len(groups) // grad_accumulation
        windows = [[] for _ in range(steps)]
        for j in range(grad_accumulation):
            for k in range(steps):
                target = k if j % 2 == 0 else steps - 1 - k
                windows[target].append(j * steps + k)
        group_order = [
            index for k in rng.permutation(steps) for index in windows[int(k)]
        ]
        group_order.extend(range(steps * grad_accumulation, len(groups)))
        plan = [pack for index in group_order for pack in groups[index]]
    if carry:
        tail = [list(pack) for pack in carry]
        while len(tail) < dp_world_size:
            splittable = [i for i, pack in enumerate(tail) if len(pack) > 1]
            if not splittable:
                break
            index = max(
                splittable, key=lambda i: sum(int(lengths[row]) for row in tail[i])
            )
            tail.append([tail[index].pop()])
        tail.extend([[] for _ in range(dp_world_size - len(tail))])
        plan.extend(tuple(tail[int(i)]) for i in rng.permutation(dp_world_size))
    return plan


def calculate_matched_group(sequences, packing_length, is_finished=True):
    """Preserve the original deterministic binpacking assignment."""
    import binpacking

    if not sequences:
        return [], []
    groups = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    if groups and not is_finished:
        return groups[:-1], groups[-1]
    return groups, []
