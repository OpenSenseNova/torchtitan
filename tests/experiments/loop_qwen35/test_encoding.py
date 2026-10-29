# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""SFT contract checks that do not require model weights or a GPU."""

import pytest
from torchtitan.experiments.loop_qwen35.encoding import (
    format_tool_schema,
    Qwen35Encoder,
)


def test_preencoded_labels_are_unshifted_and_validated():
    encoder = object.__new__(Qwen35Encoder)
    sample = dict(input_ids=[1, 2, 3], labels=[-100, 2, 3])
    assert encoder.encode(sample) == sample
    with pytest.raises(ValueError, match="align"):
        encoder.encode(dict(input_ids=[1, 2, 3], labels=[2, 3, -100]))
    with pytest.raises(ValueError, match="align"):
        encoder.encode(dict(input_ids=[1], labels=[]))


def test_tool_schema_matches_training_xml():
    schema = format_tool_schema(
        [
            dict(
                type="function",
                function=dict(
                    name="search",
                    description=" Find things. ",
                    parameters=dict(
                        type="object",
                        properties=dict(
                            query=dict(
                                type="string", description=" Query ", enum=["a", "b"]
                            )
                        ),
                        required=["query"],
                    ),
                ),
            )
        ]
    )
    assert schema == (
        "<function>\n<name>search</name>\n<description>Find things.</description>"
        "\n<parameters>\n<parameter>\n<name>query</name>\n<type>string</type>"
        '\n<description>Query</description>\n<enum>["a", "b"]</enum>\n</parameter>'
        '\n<required>["query"]</required>\n</parameters>\n</function>'
    )


@pytest.mark.parametrize("parameters", ['{"type":"object"}', {"properties": "{}"}])
def test_string_parameter_schema_retains_original_rendering(parameters):
    assert format_tool_schema([dict(name="code", parameters=parameters)]) == (
        "<function>\n<name>code</name>\n<parameters>\n</parameters>\n</function>"
    )


def test_images_require_matching_placeholders():
    encoder = object.__new__(Qwen35Encoder)
    with pytest.raises(ValueError, match="placeholders"):
        encoder._messages(
            dict(
                messages=[
                    dict(role="user", content="<image><image>"),
                    dict(role="assistant", content="answer"),
                ],
                images=["one.png"],
            )
        )


def test_partial_conversation_is_rejected():
    with pytest.raises(ValueError, match="end in an assistant"):
        object.__new__(Qwen35Encoder)._messages(
            dict(messages=[dict(role="user", content="Question")])
        )
