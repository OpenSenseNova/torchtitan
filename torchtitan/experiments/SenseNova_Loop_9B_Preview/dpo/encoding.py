# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Native Qwen3.5 preference branches and binary assistant loss masks."""

import copy
from types import SimpleNamespace

from torchtitan.experiments.SenseNova_Loop_9B_Preview.encoding import (
    Qwen35Encoder,
    normalize_messages,
)


def _branch(source):
    source = copy.deepcopy(source)
    messages = [
        {k: v for k, v in item.items() if v is not None} for item in source["messages"]
    ]
    messages = normalize_messages(messages)
    system = source.get("system")
    if messages and messages[0]["role"] == "system":
        if system is not None:
            raise ValueError("Specify system either in messages or at top level")
        system = messages.pop(0)["content"]
    images = source.get("images") or []
    if not isinstance(images, (list, tuple)):
        images = [images]
    inline_images = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text = ""
            for item in content:
                kind = item["type"]
                if kind == "text":
                    text += item["text"]
                elif kind in ("image", "image_url"):
                    value = item.get(kind)
                    if isinstance(value, dict):
                        value = value["url"]
                    inline_images.append(value)
                    text += "<image>"
                else:
                    raise ValueError("DPO supports text and static images only")
            message["content"] = text
        if message.get("loss_scale", 1) not in (0, 1):
            raise ValueError("DPO requires binary message loss_scale")
    if inline_images:
        if images:
            raise ValueError("Specify images either inline or at top level")
        images = inline_images
    if source.get("videos") or source.get("audios") or source.get("objects"):
        raise ValueError("DPO supports text/images, not video/audio/objects")
    return SimpleNamespace(
        messages=messages,
        system=system,
        images=list(images),
        tools=source.get("tools"),
        chat_template_kwargs=source.get("chat_template_kwargs") or {},
    )


class PreferenceInputs:
    @classmethod
    def from_dict(cls, source):
        source = {k: copy.deepcopy(v) for k, v in source.items() if v is not None}
        chosen = {k: v for k, v in source.items() if not k.startswith("rejected_")}
        rejected = {
            k[len("rejected_") :]: v
            for k, v in source.items()
            if k.startswith("rejected_")
        }
        response = source.get("rejected_response")
        if source.get("rejected_messages") is not None and response is not None:
            raise ValueError("Specify rejected_messages or rejected_response, not both")
        if response is not None:
            if isinstance(response, str):
                response = [dict(role="assistant", content=response)]
            if not isinstance(response, list) or any(
                m["role"] == "user" for m in response
            ):
                raise ValueError(
                    "rejected_response must be text or response messages without user turns"
                )
            last_user = max(
                (i for i, m in enumerate(chosen["messages"]) if m["role"] == "user"),
                default=-1,
            )
            rejected = {
                **chosen,
                **rejected,
                "messages": chosen["messages"][: last_user + 1] + response,
            }
        if "messages" not in rejected:
            raise ValueError("Preference data is missing the rejected branch")
        for key in ("tools", "chat_template_kwargs"):
            if rejected.get(key) is None:
                rejected[key] = chosen.get(key)
        return SimpleNamespace(chosen=_branch(chosen), rejected=_branch(rejected))


class Qwen35PreferenceEncoder(Qwen35Encoder):
    def __init__(self, model_path, loss_scale="default"):
        super().__init__(model_path)
        self.ignore_empty_think = True
        self._loss_scale = (
            loss_scale
            if "+ignore_empty_think" in loss_scale
            else loss_scale + "+ignore_empty_think"
        )
        self.template_meta = SimpleNamespace(default_system=None)

    @property
    def score_last_round(self):
        return self._loss_scale.split("+", 1)[0] == "last_round"

    @property
    def loss_scale(self):
        return SimpleNamespace(base_strategy=self._loss_scale.split("+", 1)[0])

    def encode(self, sample, *, return_length=False):
        inputs = PreferenceInputs.from_dict(sample)
        encoded = {}
        lengths = []
        for name in ("chosen", "rejected"):
            branch = vars(getattr(inputs, name))
            output = super().encode(branch)
            encoded.update({f"{name}_{key}": value for key, value in output.items()})
            lengths.append(len(output["input_ids"]))
        if return_length:
            encoded["lengths"] = lengths
        return encoded
