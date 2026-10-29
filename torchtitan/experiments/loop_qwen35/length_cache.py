# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Fingerprint and rebuild Arrow packing lengths with the training encoder."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np


def build_template(model_path):
    from torchtitan.experiments.loop_qwen35.encoding import Qwen35Encoder

    return Qwen35Encoder(model_path)


def encoding_fingerprint(datasets, model_path):
    """Bind a sidecar to ordered Arrow shards, assets, code and preprocessing."""
    root = Path(__file__).resolve().parent
    code = hashlib.sha256()
    # Qwen's vision patch reads these keys dynamically, so literal-call scanning
    # alone cannot discover them.
    env_names = {
        "IMAGE_FACTOR",
        "MIN_PIXELS",
        "MAX_PIXELS",
        "VIDEO_MIN_PIXELS",
        "VIDEO_MAX_PIXELS",
        "VIDEO_TOTAL_PIXELS",
        "MAX_RATIO",
        "FRAME_FACTOR",
        "FPS",
        "FPS_MIN_FRAMES",
        "FPS_MAX_FRAMES",
        "IMAGE_MAX_TOKEN_NUM",
        "IMAGE_MIN_TOKEN_NUM",
        "SPATIAL_MERGE_SIZE",
        "VIDEO_MAX_TOKEN_NUM",
        "VIDEO_MIN_TOKEN_NUM",
        "ROOT_IMAGE_DIR",
    }
    sources = [root / "encoding.py", root / "rendering.py"]
    sources.append(Path(__file__))
    for path in sources:
        content = path.read_bytes()
        code.update(str(path.relative_to(root)).encode())
        code.update(content)
        env_names.update(
            re.findall(r"get_env_args\(\s*['\"]([^'\"]+)", content.decode())
        )
    assets = {}
    for path in sorted(Path(model_path).iterdir()):
        if path.is_file() and path.suffix in (
            ".json",
            ".jinja",
            ".txt",
            ".model",
            ".tiktoken",
            ".py",
        ):
            assets[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shards = []
    for dataset in datasets:
        files = []
        for record in dataset.cache_files:
            path = Path(record["filename"]).resolve()
            stat = path.stat()
            files.append((str(path), stat.st_size, stat.st_mtime_ns))
        shards.append(
            dict(rows=len(dataset), fingerprint=dataset._fingerprint, files=files)
        )
    packages = {}
    for name in (
        "transformers",
        "tokenizers",
        "datasets",
        "qwen-vl-utils",
        "jinja2",
        "pillow",
        "numpy",
        "torch",
        "torchvision",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = {}
    for key in sorted({name.upper() for name in env_names}):
        value = os.environ.get(key)
        environment[key] = (
            hashlib.sha256(value.encode()).hexdigest() if value is not None else None
        )
    payload = dict(
        version=1,
        datasets=shards,
        assets=assets,
        source_sha256=code.hexdigest(),
        packages=packages,
        environment=environment,
        mode="train",
        max_length=None,
        reject_field="rejected_messages",
        shifted_sft_targets=True,
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return dict(
        sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        configuration=json.loads(canonical),
    )


def load_cache(path, fingerprint, row_count):
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        ids, lengths = cache["row_ids"], cache["lengths"]
    if metadata["fingerprint"] != fingerprint or metadata["rows"] != row_count:
        raise ValueError(
            "Packing length cache does not match the dataset/model/template/preprocessing; rebuild it"
        )
    if (
        ids.dtype != np.int64
        or lengths.dtype != np.int64
        or ids.ndim != 1
        or lengths.shape != ids.shape
    ):
        raise ValueError("Invalid packing length cache arrays")
    if (
        len(set(ids.tolist())) != len(ids)
        or (ids < 0).any()
        or (ids >= row_count).any()
        or (lengths < 0).any()
    ):
        raise ValueError("Invalid packing length cache row IDs or lengths")
    return ids, lengths


def save_cache(path, fingerprint, row_count, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(sorted(records), dtype=np.int64)
    lengths = np.asarray([records[int(index)] for index in ids], dtype=np.int64)
    metadata = json.dumps(
        dict(
            fingerprint=fingerprint,
            rows=row_count,
            encoded_rows=len(ids),
            complete=len(ids) == row_count,
        ),
        sort_keys=True,
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, row_ids=ids, lengths=lengths, metadata=metadata)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--row-indices",
        nargs="+",
        type=int,
        help="Rebuild selected rows; otherwise rebuild all rows",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100000,
        help="Rows between atomic checkpoints; larger values reduce index rewrite I/O",
    )
    args = parser.parse_args()
    if args.save_every < 1:
        parser.error("--save-every must be positive")
    from datasets import concatenate_datasets, load_from_disk

    datasets = [load_from_disk(path) for path in args.dataset]
    dataset = concatenate_datasets(datasets)
    fingerprint = encoding_fingerprint(datasets, args.model)
    records = {}
    if Path(args.output).exists():
        ids, lengths = load_cache(args.output, fingerprint, len(dataset))
        records.update(zip(ids.tolist(), lengths.tolist()))
    indices = args.row_indices if args.row_indices is not None else range(len(dataset))
    if any(index < 0 or index >= len(dataset) for index in indices):
        parser.error("Row index is outside the dataset")
    template = build_template(args.model)
    newly_encoded = 0
    for index in indices:
        if index in records:
            continue
        sample = copy.deepcopy(dataset[index])
        sample.pop("rejected_messages", None)
        encoded = template.encode(sample)
        if encoded.get("pixel_values_videos") is not None:
            raise NotImplementedError(
                "This Arrow recipe supports text/images, not video timestamp packing"
            )
        records[index] = (
            len(encoded["input_ids"])
            if any(n != -100 for n in encoded["labels"][1:])
            else 0
        )
        newly_encoded += 1
        if newly_encoded % args.save_every == 0:
            save_cache(args.output, fingerprint, len(dataset), records)
            print(
                json.dumps(
                    dict(encoded_rows=len(records), rows=len(dataset), complete=False)
                ),
                flush=True,
            )
    save_cache(args.output, fingerprint, len(dataset), records)
    print(
        json.dumps(
            dict(
                output=args.output,
                encoded_rows=len(records),
                rows=len(dataset),
                complete=len(records) == len(dataset),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
