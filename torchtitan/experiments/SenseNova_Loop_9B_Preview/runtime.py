# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve optional source checkouts and worker-local runtime state."""

import os
from pathlib import Path


def configure_worker_runtime():
    # Loop uses PP=1, so global rank 0 owns the complete training summary.
    os.environ.setdefault("LOG_RANK", "0")
    # attn_gym has its own disk cache; CUTE_DSL_CACHE_DIR alone does not configure it.
    # Keep cache keys persistent across runs and avoid cross-rank AFS lock creation.
    cute_cache = os.environ.get("CUTE_DSL_CACHE_DIR")
    if cute_cache and not os.environ.get("ATTN_GYM_CUTE_CACHE_DIR"):
        rank = os.environ.get("RANK", "0")
        os.environ["ATTN_GYM_CUTE_CACHE_DIR"] = str(
            Path(cute_cache) / "attention_gym" / f"rank_{rank}"
        )
