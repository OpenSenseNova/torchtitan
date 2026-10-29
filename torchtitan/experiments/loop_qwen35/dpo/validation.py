# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Preference branch and tool-schema validation."""

import json
from typing import Optional

_REEXPORT = "Use raw messages and structured tool schemas with the training tokenizer."


class DatasetValidationError(ValueError):
    """A data contract violation must not be skipped, even with strict=False."""


class DatasetRowValidationError(DatasetValidationError):
    """An invalid source row that eager preprocessing may drop in non-strict mode."""


def normalize_tools(tools, field="tools"):
    """Decode only the supported outer JSON wrappers, never arbitrary strings."""
    try:
        if isinstance(tools, str):
            tools = json.loads(tools)
        if tools is None:
            return []
        if not isinstance(tools, list):
            raise ValueError("expected a list")
        tools = [json.loads(tool) if isinstance(tool, str) else tool for tool in tools]
        if any(not isinstance(tool, dict) for tool in tools):
            raise ValueError("expected tool objects")
        return tools
    except (TypeError, ValueError) as e:
        raise DatasetValidationError(f"Invalid {field}: {e}. {_REEXPORT}") from e


def _normalize_row_tools(tools, field="tools"):
    try:
        return normalize_tools(tools, field)
    except DatasetValidationError as e:
        raise DatasetRowValidationError(str(e)) from e


def _check_schema(schema, path):
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise DatasetRowValidationError(
            f"{path} must be a JSON schema object, got {type(schema).__name__}. "
            f"Nested schema fields must not be JSON-encoded strings. {_REEXPORT}"
        )
    for key in (
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ):
        if key not in schema:
            continue
        children = schema[key]
        if not isinstance(children, dict):
            raise DatasetRowValidationError(
                f"{path}.{key} must be an object, got {type(children).__name__}. "
                f"{_REEXPORT}"
            )
        for name, child in children.items():
            _check_schema(child, f"{path}.{key}.{name}")
    for key in (
        "items",
        "additionalProperties",
        "additionalItems",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
    ):
        if key in schema:
            children = (
                schema[key]
                if key == "items" and isinstance(schema[key], list)
                else [schema[key]]
            )
            for i, child in enumerate(children):
                _check_schema(child, f"{path}.{key}[{i}]")
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if key in schema:
            if not isinstance(schema[key], list):
                raise DatasetRowValidationError(
                    f"{path}.{key} must be an array. {_REEXPORT}"
                )
            for i, child in enumerate(schema[key]):
                _check_schema(child, f"{path}.{key}[{i}]")


def validate_tools(tools, field="tools"):
    for i, tool in enumerate(_normalize_row_tools(tools, field)):
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            raise DatasetRowValidationError(
                f"{field}[{i}].function must be an object. {_REEXPORT}"
            )
        # Legacy agent tools can use a textual parameters description. OpenAI
        # function tools, and object schemas in either format, must stay typed.
        parameters = function.get("parameters")
        if parameters is not None and (
            "function" in tool or isinstance(parameters, dict)
        ):
            _check_schema(parameters, f"{field}[{i}].parameters")


def validate_preference_context(
    inputs, default_system: Optional[str] = None, *, loss_scale="last_round"
):
    """Compare the conditioning prefix for the selected preference objective.

    default scores all supervised assistant turns, so later user/tool observations
    may diverge. last_round compares only the final response under shared history.
    Callers without an explicit strategy retain the stricter last_round contract.
    """
    chosen, rejected = inputs.chosen, inputs.rejected
    if rejected is None:
        raise DatasetRowValidationError(
            "Preference data is missing the rejected branch."
        )
    chosen_system = default_system if chosen.system is None else chosen.system
    rejected_system = default_system if rejected.system is None else rejected.system
    if chosen_system != rejected_system:
        raise DatasetRowValidationError(
            "Preference chosen/rejected system prompts differ. Use the same shared prompt."
        )
    for branch, item in (("chosen", chosen), ("rejected", rejected)):
        validate_tools(item.tools, f"{branch}_tools")
    if _normalize_row_tools(chosen.tools) != _normalize_row_tools(rejected.tools):
        raise DatasetRowValidationError(
            "Preference chosen/rejected tools differ. Omit rejected_tools to share tools, "
            "or supply identical definitions on both branches."
        )
    if chosen.chat_template_kwargs != rejected.chat_template_kwargs:
        raise DatasetRowValidationError(
            "Preference chosen/rejected chat_template_kwargs differ."
        )

    def prompt(item, branch):
        messages = item.messages
        if loss_scale.split("+", 1)[0] == "default":
            first_response = next(
                (
                    i
                    for i, message in enumerate(messages)
                    if message["role"] in {"assistant", "tool_call"}
                    and message.get("loss") is not False
                    and message.get("loss_scale") != 0
                ),
                None,
            )
            if first_response is None:
                raise DatasetRowValidationError(
                    f"Preference {branch} has no supervised response tokens in any assistant turn."
                )
            return [
                {k: v for k, v in message.items() if v is not None}
                for message in messages[:first_response]
            ]
        last_user = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if messages[i]["role"] == "user"
            ),
            -1,
        )
        if not any(
            message["role"] in {"assistant", "tool_call"}
            and message.get("loss") is not False
            and message.get("loss_scale") != 0
            for message in messages[last_user + 1 :]
        ):
            raise DatasetRowValidationError(
                f"Preference {branch} has no supervised response tokens after the last "
                "user turn; check response messages and their loss flags."
            )
        # Shared history must also have matching supervision flags, otherwise
        # its log probabilities do not cancel in the preference comparison.
        return [
            {k: v for k, v in message.items() if v is not None}
            for message in messages[: last_user + 1]
        ]

    if prompt(chosen, "chosen") != prompt(rejected, "rejected"):
        boundary = (
            "first supervised assistant response"
            if loss_scale.split("+", 1)[0] == "default"
            else "last user response"
        )
        raise DatasetRowValidationError(
            f"Preference chosen/rejected prompts differ before the {boundary}. "
            "Use the same conditioning context for the selected loss_scale."
        )
