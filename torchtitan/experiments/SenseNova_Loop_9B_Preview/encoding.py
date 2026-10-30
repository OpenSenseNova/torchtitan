# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared Qwen3.5 encoding with explicit training supervision spans."""

import copy
import json
import os
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer


from torchtitan.experiments.SenseNova_Loop_9B_Preview.rendering import (
    format_tool_calls,
    format_tool_schema as format_tool_schema,
    SPAN_TEMPLATE,
    training_spans,
)


def normalize_messages(messages):
    """Normalize tool roles without changing supervision or tool observations."""
    result = []
    index = 0
    while index < len(messages):
        message = {
            k: copy.deepcopy(v) for k, v in messages[index].items() if v is not None
        }
        if message["role"] == "tool_response":
            message["role"] = "tool"
        if message["role"] == "tool_call":
            calls = []
            flags = (message.get("loss", True), message.get("loss_scale", 1))
            while index < len(messages) and messages[index]["role"] == "tool_call":
                item = {k: v for k, v in messages[index].items() if v is not None}
                if (item.get("loss", True), item.get("loss_scale", 1)) != flags:
                    raise ValueError(
                        "Consecutive tool calls must share their loss mask"
                    )
                calls.append(item["content"])
                index += 1
            message.update(role="assistant", content=format_tool_calls(calls))
        else:
            index += 1
            if message.get("tool_calls"):
                content = message.get("content") or ""
                message["content"] = (
                    content
                    + ("\n\n" if content.strip() else "")
                    + format_tool_calls(message.pop("tool_calls"))
                )
        if message["role"] == "tool" and not isinstance(message.get("content"), str):
            message["content"] = json.dumps(message.get("content"), ensure_ascii=False)
        result.append(message)
    return result


class Qwen35Encoder:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.chat_template = SPAN_TEMPLATE
        self.processor = None

    def _messages(self, sample):
        messages = normalize_messages(sample["messages"])
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("SFT samples must end in an assistant message")
        if sample.get("system"):
            if messages[0]["role"] == "system":
                raise ValueError("Specify system either in messages or at top level")
            messages.insert(0, {"role": "system", "content": sample["system"]})
        images = sample.get("images") or []
        images = list(images) if isinstance(images, (list, tuple)) else [images]
        if sample.get("videos") or sample.get("audios"):
            raise ValueError(
                "This recipe supports text and static images, not video/audio"
            )
        has_images = bool(images)
        tag_count = sum(
            (m.get("content") or "").count("<image>")
            for m in messages
            if isinstance(m.get("content"), str)
        )
        if images and tag_count == 0:
            first_user = next(m for m in messages if m["role"] == "user")
            first_user["content"] = "<image>" * len(images) + first_user["content"]
            tag_count = len(images)
        if images and tag_count != len(images):
            raise ValueError("Image placeholders and supplied images must match")
        index = 0
        last_user = max(
            (i for i, m in enumerate(messages) if m["role"] == "user"), default=-1
        )
        for message_index, message in enumerate(messages):
            if message.get("loss_scale", 1) not in (0, 1) or message.get(
                "loss", True
            ) not in (False, True):
                raise ValueError("Message loss masks must be binary")
            message["supervise"] = (
                bool(message.get("loss", True))
                and message.get("loss_scale", 1) != 0
                and (
                    not getattr(self, "score_last_round", False)
                    or message_index > last_user
                )
            )
            message["keep_think"] = (
                not getattr(self, "score_last_round", False)
                or message_index > last_user
            )
            content = message.get("content")
            if images and isinstance(content, str) and "<image>" in content:
                parts = content.split("<image>")
                converted = []
                for part_index, text in enumerate(parts):
                    if part_index:
                        item = images[index]
                        if isinstance(item, dict):
                            item = item.get("path") or item.get("bytes")
                        if isinstance(item, bytes):
                            import io

                            from PIL import Image

                            item = Image.open(io.BytesIO(item))
                        if (
                            isinstance(item, str)
                            and not Path(item).is_absolute()
                            and not item.startswith(
                                ("http://", "https://", "data:", "file://")
                            )
                            and os.getenv("ROOT_IMAGE_DIR")
                        ):
                            item = str(Path(os.environ["ROOT_IMAGE_DIR"]) / item)
                        converted.append({"type": "image", "image": item})
                        index += 1
                    if text:
                        converted.append({"type": "text", "text": text})
                message["content"] = converted
            elif isinstance(content, list):
                has_images |= any(
                    x.get("type") in ("image", "image_url") for x in content
                )
                if any(x.get("type") in ("video", "audio") for x in content):
                    raise ValueError("Video/audio SFT is not supported")
        return messages, has_images

    def _prepare_images(self, messages, sample):
        from qwen_vl_utils import fetch_image, vision_process

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
        patch_size = self.processor.image_processor.patch_size
        merge_size = int(os.getenv("SPATIAL_MERGE_SIZE", "2"))
        if merge_size != self.processor.image_processor.merge_size:
            raise ValueError("SPATIAL_MERGE_SIZE must match the model processor")
        factor = patch_size * merge_size
        vision_process.MAX_RATIO = int(os.getenv("MAX_RATIO", "200"))
        minimum = (
            int(os.environ["IMAGE_MIN_TOKEN_NUM"]) * factor**2
            if os.getenv("IMAGE_MIN_TOKEN_NUM")
            else int(os.getenv("MIN_PIXELS", str(4 * factor**2)))
        )
        maximum = (
            int(os.environ["IMAGE_MAX_TOKEN_NUM"]) * factor**2
            if os.getenv("IMAGE_MAX_TOKEN_NUM")
            else int(os.getenv("MAX_PIXELS", str(16384 * factor**2)))
        )
        for message in messages:
            if not isinstance(message.get("content"), list):
                continue
            for item in message["content"]:
                if item.get("type") not in ("image", "image_url"):
                    continue
                value = item.get("image", item.get("image_url"))
                if isinstance(value, dict):
                    value = value["url"]
                options = dict(image=value, min_pixels=minimum, max_pixels=maximum)
                options.update(sample.get("chat_template_kwargs") or {})
                item.clear()
                item.update(
                    type="image",
                    image=fetch_image(options, image_patch_size=patch_size),
                )

    def encode(self, sample):
        # Pre-encoded data is a first-class source, with unshifted labels.
        if sample.get("input_ids") is not None:
            ids, labels = list(sample["input_ids"]), list(sample["labels"])
            if len(ids) != len(labels) or any(
                y not in (-100, x) for x, y in zip(ids, labels)
            ):
                raise ValueError(
                    "Pre-encoded labels must align with input_ids before next-token shifting"
                )
            return {
                "input_ids": ids,
                "labels": labels,
                **{
                    k: sample[k]
                    for k in ("pixel_values", "image_grid_thw")
                    if sample.get(k) is not None
                },
            }
        if sample.get("raw_sample_json"):
            sample = json.loads(sample["raw_sample_json"])
        messages, has_images = self._messages(sample)
        tools = sample.get("tools")
        if isinstance(tools, str):
            tools = json.loads(tools)
        if isinstance(tools, dict):
            tools = [tools]
        options = dict(
            chat_template=self.chat_template,
            add_generation_prompt=False,
            tools=tools or None,
            training_spans=training_spans(
                messages,
                tools,
                self.tokenizer,
                ignore_empty_think=getattr(self, "ignore_empty_think", False),
            ),
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        if has_images:
            self._prepare_images(messages, sample)
            # Tokenizer spans precede image expansion. Expand the mask explicitly;
            # the processor's assistant mask can shift spans after image tokens.
            text_encoded = self.tokenizer.apply_chat_template(messages, **options)
            processor_options = {
                k: v for k, v in options.items() if k != "return_assistant_tokens_mask"
            }
            encoded = self.processor.apply_chat_template(
                messages,
                processor_kwargs={"return_tensors": "pt", "do_resize": False},
                **processor_options,
            )
            ids = encoded["input_ids"][0].tolist()
            expanded_ids, mask = [], []
            grids = iter(encoded["image_grid_thw"])
            image_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            for token, supervised in zip(
                text_encoded["input_ids"], text_encoded["assistant_masks"]
            ):
                count = (
                    int(next(grids).prod())
                    // self.processor.image_processor.merge_size**2
                    if token == image_id
                    else 1
                )
                expanded_ids.extend([token] * count)
                mask.extend([0 if token == image_id else supervised] * count)
            if expanded_ids != ids:
                raise ValueError(
                    "Processor expansion disagrees with tokenized image placeholders"
                )
        else:
            encoded = self.tokenizer.apply_chat_template(messages, **options)
            ids, mask = encoded["input_ids"], encoded["assistant_masks"]
        if len(ids) != len(mask):
            raise ValueError("Assistant mask and expanded multimodal tokens disagree")
        labels = [token if supervised else -100 for token, supervised in zip(ids, mask)]
        return {
            "input_ids": ids,
            "labels": labels,
            **{
                k: encoded[k]
                for k in ("pixel_values", "image_grid_thw")
                if k in encoded
            },
        }
