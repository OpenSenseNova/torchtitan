# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Offline visual geometry and model-derived packing cost estimates."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from torchtitan.experiments.SenseNova_Loop_9B_Preview.length_cache import load_cache

VISION_PLAN_VERSION = 6


def image_grid(encoded):
    if encoded.get("pixel_values_videos") is not None:
        raise NotImplementedError(
            "This Arrow recipe supports text/images, not video timestamp packing"
        )
    value = encoded.get("image_grid_thw")
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    grid = np.asarray(value)
    if not grid.size:
        return []
    if (
        grid.ndim != 2
        or grid.shape[1] != 3
        or not np.issubdtype(grid.dtype, np.integer)
        or (grid <= 0).any()
    ):
        raise ValueError("image_grid_thw must contain positive integer [t, h, w] rows")
    if (grid[:, 0] != 1).any():
        raise ValueError("Static image packing requires grid_t=1")
    return grid.tolist()


def save_vision_cache(path, fingerprint, row_count, records):
    """Atomically save exact lengths and ordered image grids from one encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(sorted(records), dtype=np.int64)
    lengths = np.asarray([records[int(i)][0] for i in ids], dtype=np.int64)
    grids = [image_grid({"image_grid_thw": records[int(i)][1]}) for i in ids]
    offsets = np.cumsum([0] + [len(g) for g in grids], dtype=np.int64)
    flat = np.asarray([g for row in grids for g in row], dtype=np.int64).reshape(-1, 3)
    metadata = json.dumps(
        dict(
            fingerprint=fingerprint,
            rows=row_count,
            encoded_rows=len(ids),
            complete=len(ids) == row_count,
            vision_schema=1,
        ),
        sort_keys=True,
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(
            handle,
            row_ids=ids,
            lengths=lengths,
            image_offsets=offsets,
            image_grids=flat,
            metadata=metadata,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_vision_cache(path, fingerprint, row_count):
    ids, lengths = load_cache(path, fingerprint, row_count)
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        if metadata.get("vision_schema") != 1:
            raise ValueError(
                "Packing cache has no image geometry; rebuild with torchtitan.experiments.SenseNova_Loop_9B_Preview.length_cache_builder "
                "into a new output file, or use --packing-vision-balance false"
            )
        offsets, grids = cache["image_offsets"], cache["image_grids"]
    if (
        offsets.dtype != np.int64
        or offsets.shape != (len(ids) + 1,)
        or offsets[0] != 0
        or (np.diff(offsets) < 0).any()
        or grids.dtype != np.int64
        or grids.ndim != 2
        or grids.shape[1] != 3
        or offsets[-1] != len(grids)
    ):
        raise ValueError("Invalid packing image geometry offsets/grids")
    image_grid({"image_grid_thw": grids})
    return ids, lengths, offsets, grids


def visual_row_costs(offsets, grids, weights):
    """Use unmerged patches; images have independent bidirectional attention."""
    patches = np.prod(grids, axis=1, dtype=np.int64)
    per_image = (
        weights["linear"] * patches
        + weights["attention"] * patches.astype(np.float64) ** 2
    )
    prefix = np.concatenate(([0.0], np.cumsum(per_image)))
    return prefix[offsets[1:]] - prefix[offsets[:-1]]


def geometry_hash(ids, offsets, grids):
    digest = hashlib.sha256()
    for values in (ids, offsets, grids):
        digest.update(values.tobytes())
    return digest.hexdigest()


def resolve_vision_weights(
    resolved, *, cp, freeze_vision, activation_checkpoint, checkpoint_policy, scale=1.0
):
    """Normalize vision FLOPs to the existing decoder length-squared units.

    This is a work estimate, not a calibrated GPU-time prediction. Vision is
    CP-replicated and executes once, outside the repeated decoder interval.
    """
    base, loop = resolved["base_model"], resolved["loop_settings"]
    repeat = sum(n * p for n, p in zip(loop["counts"], loop["probabilities"]))
    layers = sum(
        (repeat if loop["start"] <= i < loop["end"] else 1.0)
        for i, kind in enumerate(base["text_config.layer_types"])
        if kind == "full_attention"
    )
    recompute = int(activation_checkpoint and checkpoint_policy == "full")
    decoder = (
        2
        * base["text_config.num_attention_heads"]
        * base["text_config.head_dim"]
        * layers
        * (3 + recompute)
        / cp
    )
    if decoder <= 0 or not np.isfinite(scale) or scale < 0:
        raise ValueError("Invalid decoder attention cost or packing vision weight")
    dim, ff, depth = (
        base["vision_config." + key]
        for key in ("hidden_size", "intermediate_size", "depth")
    )
    vision_passes = 1 if freeze_vision else 3 + recompute
    return dict(
        linear=scale * depth * vision_passes * (8 * dim**2 + 4 * dim * ff) / decoder,
        attention=scale * depth * vision_passes * 4 * dim / decoder,
    )
