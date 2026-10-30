# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen message framing and training spans, independent of model prompt text."""

import json
from dataclasses import dataclass

# Only this small supervision wrapper is supplied to the HF tokenizer/processor.
# Model-authored tool instructions are obtained from the external model assets.
SPAN_TEMPLATE = (
    "{%- for span in training_spans -%}"
    "{%- if span.supervise -%}"
    "{% generation %}{{ span.text }}{% endgeneration %}"
    "{%- else -%}{{ span.text }}{%- endif -%}"
    "{%- endfor -%}"
)


@dataclass(frozen=True)
class TrainingSpan:
    text: str
    supervise: bool = False


def _json_object(value):
    return json.loads(value) if isinstance(value, str) else value


def _xml(tag, content):
    # The model protocol uses literal text, not escaped XML entities.
    return f"<{tag}>{content}</{tag}>"


def _schema_fields(value, names):
    """Emit required ordered fields followed by JSON-valued extension fields."""
    if not isinstance(value, dict):
        return []
    result = []
    for name in names:
        if name in value:
            item = value[name]
            text = item.strip() if name == "description" else str(item)
            result.append(_xml(name, text))
    result.extend(
        _xml(name, json.dumps(item, ensure_ascii=False))
        for name, item in value.items()
        if name not in {*names, "name", "type"}
    )
    return result


def format_tool_schema(tools):
    """Serialize function declarations in the training XML wire format."""
    functions = []
    for declaration in tools or ():
        declaration = _json_object(declaration)
        function = declaration.get("function", declaration)
        fields = [_xml("name", function["name"])]
        if "description" in function:
            fields.append(_xml("description", function["description"].strip()))
        schema = function.get("parameters", {})
        properties = schema.get("properties") if isinstance(schema, dict) else None
        parameters = []
        if isinstance(properties, dict):
            for name, specification in properties.items():
                entries = [
                    _xml("name", name),
                    *_schema_fields(specification, ("type", "description")),
                ]
                parameters.append(_xml("parameter", "\n" + "\n".join(entries) + "\n"))
        if isinstance(schema, dict):
            parameters.extend(
                _xml(name, json.dumps(value, ensure_ascii=False))
                for name, value in schema.items()
                if name not in ("type", "properties")
            )
        fields.append(
            _xml(
                "parameters",
                "\n" + "\n".join(parameters) + ("\n" if parameters else ""),
            )
        )
        fields.extend(
            _xml(name, json.dumps(value, ensure_ascii=False))
            for name, value in function.items()
            if name not in ("type", "name", "description", "parameters")
        )
        functions.append(_xml("function", "\n" + "\n".join(fields) + "\n"))
    return "\n".join(functions)


def format_tool_calls(calls):
    """Serialize calls without depending on a framework's agent template."""
    rendered = []
    for record in calls:
        record = _json_object(record)
        function = record.get("function", record)
        arguments = _json_object(function.get("arguments") or {})
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        lines = ["<tool_call>", f"<function={function['name']}>"]
        for name, value in arguments.items():
            text = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else str(value)
            )
            lines.extend((f"<parameter={name}>", text, "</parameter>"))
        lines.extend(("</function>", "</tool_call>"))
        rendered.append("\n".join(lines))
    return "\n".join(rendered)


def _content(message):
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for item in content:
        if "image" in item or "image_url" in item or item.get("type") == "image":
            if message["role"] == "system":
                raise ValueError("System message cannot contain images.")
            parts.append("<|vision_start|><|image_pad|><|vision_end|>")
        elif "text" in item:
            parts.append(item["text"])
        else:
            raise ValueError("Unexpected item type in content.")
    return "".join(parts).strip()


def _tool_header(tokenizer, system, tools):
    """Ask the selected model for its own prompt; replace only the tool payload."""
    # A sentinel user turn lets us isolate the system frame without copying
    # model-authored instructions into the repository.
    sentinel = "torchtitan_tool_prompt_boundary"
    messages = [{"role": "user", "content": sentinel}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    user_frame = f"<|im_start|>user\n{sentinel}<|im_end|>\n"
    if not text.endswith(user_frame):
        raise ValueError("The model chat template must use Qwen message framing")
    prefix = text[: -len(user_frame)]
    opening, closing = "<tools>\n", "\n</tools>"
    start = prefix.find(opening)
    end = prefix.find(closing, start + len(opening))
    if start < 0 or end < 0:
        raise ValueError("The model chat template must provide a tools section")
    return prefix[: start + len(opening)] + format_tool_schema(tools) + prefix[end:]


def training_spans(messages, tools, tokenizer, *, ignore_empty_think=False):
    """Frame normalized messages and mark exactly the assistant target text."""
    if not messages:
        raise ValueError("No messages provided.")
    if not any(
        message["role"] == "user"
        and not (
            _content(message).startswith("<tool_response>")
            and _content(message).endswith("</tool_response>")
        )
        for message in messages
    ):
        raise ValueError("No user query found in messages.")
    first_is_system = messages[0]["role"] == "system"
    system = _content(messages[0]) if first_is_system else None
    spans = []
    if tools:
        spans.append(TrainingSpan(_tool_header(tokenizer, system, tools)))
    elif system is not None:
        spans.append(TrainingSpan(f"<|im_start|>system\n{system}<|im_end|>\n"))
    turns = messages[1:] if first_is_system else messages
    for index, message in enumerate(turns):
        role = message["role"]
        content = _content(message)
        if role == "assistant":
            reasoning = message.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = ""
                if "</think>" in content:
                    first_end = content.index("</think>")
                    opening = content.rfind("<think>", 0, first_end)
                    reasoning = content[
                        opening + len("<think>") if opening >= 0 else 0 : first_end
                    ]
                    content = content[
                        content.rindex("</think>") + len("</think>") :
                    ].lstrip("\n")
            reasoning = reasoning.strip()
            spans.append(TrainingSpan("<|im_start|>assistant\n"))
            supervised = bool(message.get("supervise"))
            thinking = (
                f"<think>\n{reasoning}\n</think>\n\n"
                if message.get("keep_think")
                else ""
            )
            body = thinking + content + "<|im_end|>\n"
            if supervised and ignore_empty_think and not reasoning:
                # Empty reasoning is context, never a DPO target.
                empty = "<think>\n\n</think>\n\n"
                spans.append(TrainingSpan(empty))
                body = body[len(empty) :]
            spans.append(TrainingSpan(body, supervised))
        elif role == "tool":
            if index == 0 or turns[index - 1]["role"] != "tool":
                spans.append(TrainingSpan("<|im_start|>user"))
            spans.append(
                TrainingSpan("\n" + _xml("tool_response", "\n" + content + "\n"))
            )
            if index + 1 == len(turns) or turns[index + 1]["role"] != "tool":
                spans.append(TrainingSpan("<|im_end|>\n"))
        elif role == "user":
            spans.append(TrainingSpan(f"<|im_start|>user\n{content}<|im_end|>\n"))
        elif role == "system":
            raise ValueError("System message must be at the beginning.")
        else:
            raise ValueError("Unexpected message role.")
    return spans
