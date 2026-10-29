# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Exact offline/online encoding contracts, including causal label boundaries."""

import copy
import sys
from types import SimpleNamespace

import pytest
from datasets import Dataset, load_from_disk

from torchtitan.experiments.loop_qwen35 import length_cache, length_cache_builder
from torchtitan.experiments.loop_qwen35.data import ArrowDataLoader
from torchtitan.experiments.loop_qwen35.dpo.data import (
    build_preference_template,
    encode_pair,
)
from torchtitan.experiments.loop_qwen35.packing_vision import (
    load_vision_cache,
    save_vision_cache,
)


def rows():
    return [
        dict(
            messages=[
                dict(role="user", content="Unicode?"),
                dict(role="assistant", content="q\u0304_t \u0628\u0650\u0645\u0627"),
            ],
            rejected_response="Unused SFT answer",
            lengths=1,
        ),
        dict(
            messages=[
                dict(role="user", content="Hello"),
                dict(role="assistant", content="history.", loss=False),
                dict(role="user", content="question?"),
                dict(role="assistant", content="<think></think>good answer"),
            ],
            lengths=1,
        ),
        dict(
            messages=[
                dict(role="user", content="A warning"),
                dict(role="assistant", content="\u26a0\ufe0f\\n1. good"),
            ],
            lengths=1,
        ),
    ]


def training_loader(template, source, *, strategy="greedy", cache=""):
    return ArrowDataLoader(
        ArrowDataLoader.Config(
            paths=[str(source)],
            model_path=template.model_path,
            packing=strategy == "balanced",
            packing_strategy=strategy,
            packing_length_cache=str(cache),
        ),
        dp_world_size=1,
        dp_rank=0,
        tokenizer=SimpleNamespace(pad_id=0, image_id=999, video_id=998),
        max_context_length=2048,
        num_tokens_per_microbatch=2048,
    )


@pytest.mark.parametrize("strategy", ["greedy", "balanced"])
def test_serial_and_worker_encoding_match_training(
    template, tmp_path, monkeypatch, strategy
):
    source = tmp_path / "arrow"
    Dataset.from_list(rows()).save_to_disk(str(source))
    offline = length_cache.build_template(template.model_path)
    encoded = []
    encode = offline.encode

    def record(sample):
        result = encode(sample)
        encoded.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(offline, "encode", record)
    monkeypatch.setattr(length_cache, "build_template", lambda _: offline)
    cache = tmp_path / "lengths.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "length_cache",
            "--dataset",
            str(source),
            "--model",
            template.model_path,
            "--output",
            str(cache),
        ],
    )
    length_cache.main()
    assert len(encoded) == 3

    length_cache_builder._initialize([str(source)], template.model_path)
    worker = length_cache_builder._TEMPLATE
    worker_encode = worker.encode
    worker_results = []

    def record_worker(sample):
        result = worker_encode(sample)
        worker_results.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(worker, "encode", record_worker)
    lengths = [length_cache_builder._encode(i)[1][0] for i in range(3)]
    assert worker_results == encoded
    assert lengths == [len(r["input_ids"]) for r in encoded]

    expected = {tuple(r["input_ids"]): r for r in encoded}
    loader = training_loader(
        template,
        source,
        strategy=strategy,
        cache=cache if strategy == "balanced" else "",
    )
    seen = []
    try:
        for _ in range(1 if strategy == "balanced" else 3):
            batch = next(loader)
            offset = 0
            for length in batch.document_lengths:
                tokens = tuple(batch.input[offset : offset + length].tolist())
                reference = expected[tokens]
                assert batch.labels[offset : offset + length].tolist() == (
                    reference["labels"][1:] + [-100]
                )
                assert batch.positions[offset : offset + length].tolist() == list(
                    range(length)
                )
                assert not batch.padding_mask[offset : offset + length].any()
                seen.append(tokens)
                offset += length
            assert batch.padding_mask[offset:].all()
            assert (batch.labels[offset:] == -100).all()
        assert sorted(seen) == sorted(expected)
    finally:
        loader.close()


@pytest.mark.parametrize("loss_scale", ["default", "last_round"])
def test_preference_offline_export_matches_training(template, loss_scale):
    sample = dict(
        messages=[
            dict(role="user", content="Hello?"),
            dict(role="assistant", content="history."),
            dict(role="user", content="question?"),
            dict(role="assistant", content="<think></think>q\u0304_t good"),
        ],
        rejected_response="\u0628\u0650\u0645\u0627 bad",
    )
    offline = build_preference_template(template.model_path, loss_scale)
    online = build_preference_template(template.model_path, loss_scale)
    expected = offline.encode(copy.deepcopy(sample), return_length=True)
    packed, exported = encode_pair(
        online,
        sample,
        "fixture:0",
        max_length=2048,
        token_budget=4096,
        return_encoding=True,
    )
    assert exported == expected
    for branch, row in zip(("chosen", "rejected"), packed):
        assert row["input_ids"].tolist() == expected[f"{branch}_input_ids"]
        assert row["labels"].tolist() == expected[f"{branch}_labels"][1:] + [-100]


@pytest.mark.parametrize("cache_type", ["length", "vision"])
@pytest.mark.parametrize(
    "package",
    [
        "transformers",
        "tokenizers",
        "datasets",
        "qwen-vl-utils",
        "jinja2",
        "pillow",
        "numpy",
        "torch",
        "torchvision",
    ],
)
def test_cache_rejects_changed_encoding_dependency(
    template, tmp_path, monkeypatch, package, cache_type
):
    source = tmp_path / "arrow"
    Dataset.from_list(rows()[:1]).save_to_disk(str(source))
    datasets = [load_from_disk(str(source))]
    original = length_cache.encoding_fingerprint(datasets, template.model_path)
    cache = tmp_path / "lengths.npz"
    if cache_type == "length":
        length_cache.save_cache(cache, original, 1, {0: 20})
        load = length_cache.load_cache
    else:
        save_vision_cache(cache, original, 1, {0: (20, [])})
        load = load_vision_cache
    version = length_cache.importlib.metadata.version
    monkeypatch.setattr(
        length_cache.importlib.metadata,
        "version",
        lambda name: version(name) + ".changed" if name == package else version(name),
    )
    changed = length_cache.encoding_fingerprint(datasets, template.model_path)
    with pytest.raises(ValueError, match="does not match"):
        load(cache, changed, 1)
    with pytest.raises(ValueError, match="does not match"):
        training_loader(template, source, strategy="balanced", cache=cache)


def test_no_shifted_targets_are_invalid_offline_and_online(template, tmp_path):
    source = tmp_path / "arrow"
    Dataset.from_list(
        [
            dict(input_ids=[10, 11], labels=[10, -100]),
            dict(input_ids=[10, 12], labels=[-100, 12]),
        ]
    ).save_to_disk(str(source))
    length_cache_builder._initialize([str(source)], template.model_path)
    assert length_cache_builder._encode(0)[1][0] == 0
    assert length_cache_builder._encode(1)[1][0] == 2
    loader = training_loader(template, source)
    try:
        for _ in range(2):
            batch = next(loader)
            assert batch.document_lengths == (2,)
            assert batch.input[:2].tolist() == [10, 12]
            assert batch.labels[:2].tolist() == [12, -100]
        assert loader.skipped > 0
    finally:
        loader.close()
