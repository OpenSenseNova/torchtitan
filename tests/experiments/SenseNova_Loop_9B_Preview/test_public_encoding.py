# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Independent public data contracts, shared across the two objectives."""

import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from torchtitan.experiments.SenseNova_Loop_9B_Preview.encoding import Qwen35Encoder
from torchtitan.experiments.SenseNova_Loop_9B_Preview.data import (
    load_dataset_sources,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.model_config import (
    resolve_hf_path,
)


def test_tool_roles_score_the_xml_call_in_both_objectives(template):
    messages = [
        dict(role="user", content="Calculate"),
        dict(role="tool_call", content={"name": "calc", "arguments": {"x": 2}}),
        dict(role="tool_response", content="2"),
        dict(role="assistant", content="The answer is 2"),
    ]
    expected = copy.deepcopy(messages)
    expected[1] = dict(
        role="assistant",
        content="<tool_call>\n<function=calc>\n<parameter=x>\n2\n</parameter>\n</function>\n</tool_call>",
    )
    expected[2]["role"] = "tool"
    sft = Qwen35Encoder(template.model_path)
    assert sft.encode(dict(messages=messages)) == sft.encode(dict(messages=expected))
    template._loss_scale = "default+ignore_empty_think"
    actual = template.encode(dict(messages=messages, rejected_response="Wrong"))
    manual = template.encode(dict(messages=expected, rejected_response="Wrong"))
    assert actual == manual
    supervised = template.tokenizer.decode(
        [x for x in actual["chosen_labels"] if x != -100]
    )
    assert "<function=calc>" in supervised
    assert "<tool_response>" not in supervised


@pytest.mark.parametrize("mask", [False, 0])
def test_sft_message_masks_keep_context_but_remove_targets(template, mask):
    sft = Qwen35Encoder(template.model_path)
    source = dict(
        messages=[
            dict(role="user", content="Hello"),
            dict(role="assistant", content="history.", loss=mask),
            dict(role="user", content="question?"),
            dict(role="assistant", content="good answer"),
        ]
    )
    encoded = sft.encode(source)
    all_text = sft.tokenizer.decode(encoded["input_ids"])
    supervised = sft.tokenizer.decode([x for x in encoded["labels"] if x != -100])
    assert "history." in all_text and "history." not in supervised
    assert "question?" not in supervised and "good answer" in supervised
    source["messages"][1].pop("loss")
    source["messages"][1]["loss_scale"] = 0
    assert encoded == sft.encode(source)
    source["messages"][1]["loss_scale"] = 0.5
    with pytest.raises(ValueError, match="binary"):
        sft.encode(source)


def test_raw_jsonl_and_arrow_use_the_same_sft_encoder(template, tmp_path):
    source = dict(
        messages=[
            dict(role="user", content="Hello"),
            dict(role="assistant", content="good"),
        ]
    )
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(source) + "\n")
    json_dataset = load_dataset_sources([str(path)])[0]
    arrow = tmp_path / "arrow"
    json_dataset.save_to_disk(str(arrow))
    arrow_dataset = load_dataset_sources([str(arrow)])[0]
    encoder = Qwen35Encoder(template.model_path)
    assert encoder.encode(json_dataset[0]) == encoder.encode(arrow_dataset[0])


def test_hf_paths_use_local_assets_before_downloading(tmp_path):
    assert resolve_hf_path(str(tmp_path)) == str(tmp_path.resolve())
    with pytest.raises(FileNotFoundError):
        resolve_hf_path(str(tmp_path / "missing"))
    with patch(
        "huggingface_hub.snapshot_download", return_value="/cache/snapshot"
    ) as download:
        assert resolve_hf_path("Qwen/Qwen3.5-9B") == "/cache/snapshot"
    assert download.call_args.kwargs["repo_id"] == "Qwen/Qwen3.5-9B"
    assert "*.py" not in download.call_args.kwargs["allow_patterns"]


def test_nullable_arrow_tool_fields_keep_default_masks(template):
    row = dict(
        messages=[
            dict(role="user", content="Calculate"),
            dict(
                role="tool_call",
                content={"name": "calc", "arguments": {"x": 2}},
                loss=None,
                loss_scale=None,
            ),
            dict(role="tool_response", content="2"),
            dict(role="assistant", content="The answer is 2"),
        ]
    )
    clean = copy.deepcopy(row)
    clean["messages"][1].pop("loss")
    clean["messages"][1].pop("loss_scale")
    encoder = Qwen35Encoder(template.model_path)
    assert encoder.encode(row) == encoder.encode(clean)


@pytest.mark.parametrize(
    "change", ["data", "image_environment", "tokenizer", "model_template"]
)
def test_greedy_resume_rejects_changed_encoding_contract(
    template, tmp_path, monkeypatch, change
):
    from types import SimpleNamespace
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.data import (
        ArrowDataLoader,
    )

    path = tmp_path / "data.jsonl"
    rows = [
        dict(
            messages=[
                dict(role="user", content="Hello"),
                dict(role="assistant", content=answer),
            ]
        )
        for answer in ("good", "bad")
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def loader():
        return ArrowDataLoader(
            ArrowDataLoader.Config(paths=[str(path)], model_path=template.model_path),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=SimpleNamespace(pad_id=0, image_id=999, video_id=998),
            max_context_length=128,
            num_tokens_per_microbatch=128,
        )

    original = loader()
    state = original.state_dict()
    restored = loader()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    original.close()
    restored.close()
    if change == "data":
        rows[0]["messages"][-1]["content"] = "changed answer"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif change == "image_environment":
        monkeypatch.setenv("IMAGE_MAX_TOKEN_NUM", "4")
    elif change == "model_template":
        asset = Path(template.model_path) / "chat_template.jinja"
        asset.write_text(asset.read_text() + "\n")
    else:
        asset = Path(template.model_path) / "tokenizer_config.json"
        content = json.loads(asset.read_text())
        content["model_max_length"] = 2048
        asset.write_text(json.dumps(content))
    changed = loader()
    try:
        with pytest.raises(ValueError, match="encoding_fingerprint"):
            changed.load_state_dict(state)
    finally:
        changed.close()


@pytest.mark.parametrize(
    "text", ["q\u0304_t", "\u26a0\ufe0f\\n1.", "\u0628\u0650\u0645\u0627"]
)
def test_qwen35_legacy_tokenizer_metadata_preserves_model_rules(tmp_path, text):
    from tokenizers import Tokenizer, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

    corpus = ["q\u0304_t", "\u26a0\ufe0f\\n1.", "\u0628\u0650\u0645\u0627"]
    backend = Qwen3_5Tokenizer().backend_tokenizer
    backend.train_from_iterator(
        corpus * 20,
        trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    # Original Qwen3.5 assets retain this legacy tokenizer class name.
    config["tokenizer_class"] = "Qwen2Tokenizer"
    config_path.write_text(json.dumps(config))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

    canonical = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    encoder = Qwen35Encoder(str(tmp_path))
    assert (
        encoder.tokenizer.encode(text, add_special_tokens=False)
        == canonical.encode(text, add_special_tokens=False).ids
    )
