# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Parallel, resumable preprocessing of exact training-encoder lengths."""

import argparse
import copy
import json
import multiprocessing as mp
from pathlib import Path

from torchtitan.experiments.SenseNova_Loop_9B_Preview.length_cache import (
    build_template,
    encoding_fingerprint,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.packing_vision import (
    image_grid,
    load_vision_cache,
    save_vision_cache,
)

_DATASET = _TEMPLATE = _INITIALIZATION_ERROR = None


def _initialize(paths, model):
    from datasets import concatenate_datasets, load_from_disk

    global _DATASET, _TEMPLATE, _INITIALIZATION_ERROR
    _INITIALIZATION_ERROR = None
    try:
        _DATASET = concatenate_datasets([load_from_disk(path) for path in paths])
        _TEMPLATE = build_template(model)
    except Exception as error:
        # Pool initializers otherwise respawn failed workers indefinitely.
        # Transport the failure through the first task to the parent instead.
        _INITIALIZATION_ERROR = error


def _encode(index):
    if _INITIALIZATION_ERROR is not None:
        raise RuntimeError(
            "Encoder worker initialization failed"
        ) from _INITIALIZATION_ERROR
    sample = copy.deepcopy(_DATASET[index])
    sample.pop("rejected_messages", None)
    encoded = _TEMPLATE.encode(sample)
    if encoded.get("pixel_values_videos") is not None:
        raise NotImplementedError(
            "This Arrow recipe supports text/images, not video timestamp packing"
        )
    length = (
        len(encoded["input_ids"])
        if any(n != -100 for n in encoded["labels"][1:])
        else 0
    )
    return index, (length, image_grid(encoded))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Require an existing complete matching cache",
    )
    args = parser.parse_args()
    if min(args.workers, args.save_every) < 1:
        parser.error("workers and save-every must be positive")
    from datasets import load_from_disk

    datasets = [load_from_disk(path) for path in args.dataset]
    fingerprint = encoding_fingerprint(datasets, args.model)
    total = sum(map(len, datasets))
    records = {}
    if Path(args.output).exists():
        ids, lengths, offsets, grids = load_vision_cache(
            args.output, fingerprint, total
        )
        records.update(
            (int(index), (int(lengths[k]), grids[offsets[k] : offsets[k + 1]].tolist()))
            for k, index in enumerate(ids)
        )
    if args.verify_only or len(records) == total:
        if len(records) != total:
            parser.error(
                f"Cache is incomplete: {len(records)}/{total}; finish preprocessing before training"
            )
    else:
        indices = (i for i in range(total) if i not in records)
        if args.workers == 1:
            _initialize(args.dataset, args.model)
            results = map(_encode, indices)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(
                args.workers,
                initializer=_initialize,
                initargs=(args.dataset, args.model),
            )
            results = pool.imap(_encode, indices, chunksize=16)
        try:
            for count, (index, record) in enumerate(results, 1):
                records[index] = record
                if count % args.save_every == 0:
                    save_vision_cache(args.output, fingerprint, total, records)
                    print(
                        json.dumps(
                            dict(encoded_rows=len(records), rows=total, complete=False)
                        ),
                        flush=True,
                    )
        except BaseException:
            if pool is not None:
                pool.terminate()
            raise
        else:
            if pool is not None:
                pool.close()
        finally:
            if pool is not None:
                pool.join()
            # Retain successful rows on an interrupted/failed pass; never mark a
            # partial cache complete or continue training from a failed command.
            save_vision_cache(args.output, fingerprint, total, records)
    print(
        json.dumps(
            dict(
                output=args.output,
                encoded_rows=len(records),
                rows=total,
                complete=len(records) == total,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
