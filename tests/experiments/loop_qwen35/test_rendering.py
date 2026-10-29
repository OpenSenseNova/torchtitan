# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Behavior contracts for model-owned prompts and independent training spans."""

import pytest

from torchtitan.experiments.loop_qwen35.encoding import Qwen35Encoder
from torchtitan.experiments.loop_qwen35.rendering import (
    format_tool_calls,
    format_tool_schema,
)


def _sample(**kwargs):
    return dict(
        messages=[
            dict(role="user", content="question"),
            dict(role="assistant", content="answer"),
        ],
        **kwargs,
    )


def test_tool_instructions_come_from_the_selected_model(template):
    encoder = Qwen35Encoder(template.model_path)
    encoder.tokenizer.chat_template = encoder.tokenizer.chat_template.replace(
        "Fixture tool instructions", "Model-specific policy A"
    )
    encoded = encoder.encode(_sample(tools=[dict(name="lookup")]))
    text = encoder.tokenizer.decode(encoded["input_ids"])
    targets = encoder.tokenizer.decode([x for x in encoded["labels"] if x != -100])
    assert "Model-specific policy A" in text
    assert "<function>\n<name>lookup</name>" in text
    assert "Model-specific policy A" not in targets
    assert "<function>" not in targets
    assert "answer" in targets


def test_text_does_not_require_an_external_tool_prompt(template):
    encoder = Qwen35Encoder(template.model_path)
    encoder.tokenizer.chat_template = None
    encoded = encoder.encode(_sample())
    assert "answer" in encoder.tokenizer.decode(encoded["input_ids"])
    with pytest.raises(ValueError, match="chat template|chat_template"):
        encoder.encode(_sample(tools=[dict(name="lookup")]))


@pytest.mark.parametrize(
    "prompt",
    ["unsupported", "<|im_start|>user\ntorchtitan_tool_prompt_boundary<|im_end|>\n"],
)
def test_incompatible_model_tool_prompt_fails_explicitly(template, prompt):
    encoder = Qwen35Encoder(template.model_path)
    encoder.tokenizer.chat_template = prompt
    with pytest.raises(ValueError, match="Qwen message framing|tools section"):
        encoder.encode(_sample(tools=[dict(name="lookup")]))


def test_tool_call_payload_uses_protocol_types_and_literal_text():
    calls = [
        {
            "function": {
                "name": "lookup",
                "arguments": {"q": "a&b", "data": {"x": [1, None]}, "flag": True},
            }
        },
        {"name": "finish", "arguments": "{}"},
    ]
    assert format_tool_calls(calls) == (
        "<tool_call>\n<function=lookup>\n"
        "<parameter=q>\na&b\n</parameter>\n"
        '<parameter=data>\n{"x": [1, null]}\n</parameter>\n'
        "<parameter=flag>\nTrue\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=finish>\n</function>\n</tool_call>"
    )


def test_schema_extensions_keep_order_and_json_values():
    schema = dict(
        name="lookup",
        description=" Search. ",
        parameters=dict(
            type="object",
            properties=dict(
                q=dict(type="string", description=" Query. ", enum=["a", "b"])
            ),
            required=["q"],
            additionalProperties=False,
        ),
        strict=True,
    )
    assert format_tool_schema([schema]) == (
        "<function>\n<name>lookup</name>\n<description>Search.</description>\n"
        "<parameters>\n<parameter>\n<name>q</name>\n<type>string</type>\n"
        '<description>Query.</description>\n<enum>["a", "b"]</enum>\n'
        '</parameter>\n<required>["q"]</required>\n'
        "<additionalProperties>false</additionalProperties>\n</parameters>\n"
        "<strict>true</strict>\n</function>"
    )


def test_empty_thinking_and_history_have_the_expected_dpo_masks(template):
    template._loss_scale = "last_round+ignore_empty_think"
    pair = _sample(rejected_response="wrong")
    pair["messages"][:0] = [
        dict(role="user", content="history question"),
        dict(
            role="assistant",
            content="history answer",
            reasoning_content="old reasoning",
        ),
    ]
    encoded = template.encode(pair)
    text = template.tokenizer.decode(encoded["chosen_input_ids"])
    targets = template.tokenizer.decode(
        [x for x in encoded["chosen_labels"] if x != -100]
    )
    assert "history answer" in text
    assert "old reasoning" not in text
    assert "history answer" not in targets
    assert "<think>" not in targets
    assert targets == "answer<|im_end|>\n"


@pytest.mark.parametrize(
    "content, expected",
    [
        ("<think>reason</think> answer", "<think>\nreason\n</think>\n\n answer"),
        (
            "<think>first</think> ignored <think>second</think>\nfinal",
            "<think>\nfirst\n</think>\n\nfinal",
        ),
    ],
)
def test_embedded_reasoning_preserves_answer_whitespace(template, content, expected):
    encoder = Qwen35Encoder(template.model_path)
    sample = _sample()
    sample["messages"][-1]["content"] = content
    encoded = encoder.encode(sample)
    targets = encoder.tokenizer.decode([x for x in encoded["labels"] if x != -100])
    assert targets == expected + "<|im_end|>\n"


@pytest.mark.parametrize("ignore_empty_think", [False, True])
def test_image_processor_preserves_tensors_and_supervision(
    template, ignore_empty_think
):
    import torch
    from PIL import Image
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    encoder = Qwen35Encoder(template.model_path)
    encoder.ignore_empty_think = ignore_empty_think
    encoder.tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
            ]
        }
    )
    encoder.processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2),
        tokenizer=encoder.tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=encoder.tokenizer.chat_template,
    )
    encoded = encoder.encode(_sample(images=[Image.new("RGB", (64, 64))]))
    assert isinstance(encoded["pixel_values"], torch.Tensor)
    assert encoded["image_grid_thw"].tolist() == [[1, 4, 4]]
    image_id = encoder.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_positions = [
        i for i, token in enumerate(encoded["input_ids"]) if token == image_id
    ]
    assert len(image_positions) == 4
    assert all(encoded["labels"][i] == -100 for i in image_positions)
    targets = encoder.tokenizer.decode([x for x in encoded["labels"] if x != -100])
    expected = "answer<|im_end|>\n"
    if not ignore_empty_think:
        expected = "<think>\n\n</think>\n\n" + expected
    assert targets == expected
