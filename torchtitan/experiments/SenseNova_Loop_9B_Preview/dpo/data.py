# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Raw Arrow preference pairs with explicit optimizer-step ownership."""

import copy
import hashlib
import json
import numpy as np
import torch
from bisect import bisect_right
from dataclasses import dataclass, field
from datasets import concatenate_datasets
from pathlib import Path

from torchtitan.experiments.SenseNova_Loop_9B_Preview.data import (
    ArrowDataLoader,
    load_dataset_sources,
)
from torchtitan.experiments.SenseNova_Loop_9B_Preview.length_cache import (
    encoding_fingerprint,
)
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import (
    DatasetBuildContext,
    TokenizedTrainingMicrobatch,
)
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator

PLAN_VERSION = 2


def build_preference_template(model_path, loss_scale="default"):
    from .encoding import Qwen35PreferenceEncoder

    return Qwen35PreferenceEncoder(model_path, loss_scale)


@dataclass(kw_only=True, slots=True)
class PreferenceMicrobatch(TokenizedTrainingMicrobatch):
    sequence_ids: torch.Tensor
    pair_ids: tuple[str, ...]
    pair_valid: torch.Tensor
    document_lengths: tuple[int, ...] = ()
    completion_counts: torch.Tensor

    @property
    def num_valid_pairs(self):
        return int(self.pair_valid.sum())


def encode_pair(
    template, source, source_id, *, max_length, token_budget, return_encoding=False
):
    """Validate and shift each document once; optionally reuse the raw encoding for export."""
    from .validation import validate_preference_context
    from .encoding import PreferenceInputs as TemplateInputs

    try:
        if source.get("raw_sample_json"):
            source = json.loads(source["raw_sample_json"])
        source = {key: value for key, value in source.items() if value is not None}
        for key in ("messages", "rejected_messages", "rejected_response"):
            messages = source.get(key)
            if isinstance(messages, list):
                if any(
                    message.get("loss_scale", 1) not in (None, 0, 1)
                    for message in messages
                ):
                    raise ValueError("DPO requires binary message loss_scale")
        for key in ("videos", "audios", "rejected_videos", "rejected_audios"):
            if source.get(key):
                raise ValueError("DPO supports text/images, not video or audio")
        inputs = TemplateInputs.from_dict(copy.deepcopy(source))
        validate_preference_context(
            inputs,
            template.template_meta.default_system,
            loss_scale=template.loss_scale.base_strategy,
        )
        if inputs.chosen.messages == inputs.rejected.messages:
            raise ValueError("chosen and rejected responses are identical")
        # Compare source identities before a processor opens image assets.
        if inputs.chosen.images != inputs.rejected.images:
            raise ValueError("chosen/rejected images or image order differ")
        encoded = template.encode(copy.deepcopy(source), return_length=return_encoding)
        rows = []
        for branch in ("chosen", "rejected"):
            tokens = torch.as_tensor(encoded[f"{branch}_input_ids"], dtype=torch.long)
            labels = torch.as_tensor(encoded[f"{branch}_labels"], dtype=torch.long)
            scale = encoded.get(f"{branch}_loss_scale")
            if scale is not None and any(value not in (0, 1) for value in scale):
                raise ValueError(
                    "DPO requires a binary completion mask, not weighted loss_scale"
                )
            if scale is not None:
                labels = labels.masked_fill(torch.as_tensor(scale) == 0, -100)
            shifted = torch.cat((labels[1:], labels.new_full((1,), -100)))
            if not (shifted != -100).any():
                raise ValueError(
                    f"{branch} has zero completion tokens after causal shift"
                )
            row = dict(
                input_ids=tokens, labels=shifted, positions=torch.arange(tokens.numel())
            )
            for key in ("pixel_values", "image_grid_thw"):
                value = encoded.get(f"{branch}_{key}")
                if value is not None:
                    row[key] = value
            rows.append(row)
        lengths = [row["input_ids"].numel() for row in rows]
        if max(lengths) > max_length or sum(lengths) > token_budget:
            raise ValueError(
                f"pair lengths={lengths}, max_length={max_length}, token_budget={token_budget}; "
                "DPO never truncates either response"
            )
        if torch.equal(rows[0]["input_ids"], rows[1]["input_ids"]):
            raise ValueError("chosen/rejected tokenize to identical sequences")
        for key in ("pixel_values", "image_grid_thw"):
            a, b = rows[0].get(key), rows[1].get(key)
            if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                raise ValueError(f"encoded chosen/rejected {key} differ")
        return (rows, encoded) if return_encoding else rows
    except (ValueError, AssertionError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid preference pair {source_id}: {error}") from error


def pack_pairs(pairs, *, token_budget, max_pairs, packing):
    """Whole pairs only; binning does not change logical batch membership."""
    groups, group, used = [], [], 0
    for item in pairs:
        length = sum(row["input_ids"].numel() for row in item[1])
        if group and (
            not packing or len(group) == max_pairs or used + length > token_budget
        ):
            groups.append(group)
            group, used = [], 0
        group.append(item)
        used += length
    if group:
        groups.append(group)
    return groups


class PreferenceDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        paths: list[str] = field(default_factory=list)
        model_path: str = ""
        loss_scale: str = "default"
        packing: bool = False
        micro_batch_size: int = 1
        global_batch_size: int = 1
        seed: int = 42

    def __init__(
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        tokenizer,
        max_context_length,
        num_tokens_per_microbatch,
        **kwargs,
    ):
        import grain.python as grain

        from torchtitan.observability.logging import init_logger

        self.config = config
        datasets = load_dataset_sources(config.paths)
        self.dataset = concatenate_datasets(datasets)
        if not len(self.dataset) or not {"messages", "raw_sample_json"}.intersection(
            self.dataset.column_names
        ):
            raise ValueError(
                "DPO requires nonempty raw Arrow conversations with rejected responses"
            )
        self.dataset_offsets = np.cumsum(
            [0] + [len(dataset) for dataset in datasets]
        ).tolist()
        self.dp_world_size, self.dp_rank = dp_world_size, dp_rank
        if config.global_batch_size % (config.micro_batch_size * dp_world_size):
            raise ValueError("DPO GBS must be divisible by MBS * DP")
        self.token_budget, self.max_length = (
            num_tokens_per_microbatch,
            max_context_length,
        )
        self.template = build_preference_template(config.model_path, config.loss_scale)
        init_logger()
        fingerprint = encoding_fingerprint(datasets, config.model_path)
        encoder_hash = hashlib.sha256()
        for name in ("encoding.py", "validation.py", "data.py"):
            encoder_hash.update(Path(__file__).with_name(name).read_bytes())
        self.contract = dict(
            preference_encoder=encoder_hash.hexdigest(),
            encoding=fingerprint,
            mode="rlhf",
            loss_scale=self.template._loss_scale,
            plan=PLAN_VERSION,
            seed=config.seed,
            mbs=config.micro_batch_size,
            gbs=config.global_batch_size,
            dp=dp_world_size,
            packing=config.packing,
            budget=self.token_budget,
            max_length=self.max_length,
        )
        self.fingerprint = hashlib.sha256(
            json.dumps(self.contract, sort_keys=True).encode()
        ).hexdigest()
        context = DatasetBuildContext(
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            num_tokens_per_microbatch=num_tokens_per_microbatch,
            read_options=grain.ReadOptions(),
        )
        self.collator = MultiModalCollator.Config(build_mrope_positions=True).build(
            context=context
        )
        self.epoch = self.cursor = 0
        self._set_order()

    def _set_order(self):
        self.order = np.random.default_rng(self.config.seed + self.epoch).permutation(
            len(self.dataset)
        )

    def source_id(self, index):
        source = bisect_right(self.dataset_offsets, index) - 1
        return f"{source}:{index - self.dataset_offsets[source]}"

    def next_step(self):
        if self.cursor == len(self.dataset):
            self.epoch += 1
            self.cursor = 0
            self._set_order()
        indices = self.order[self.cursor : self.cursor + self.config.global_batch_size]
        pairs = []
        for position, index in enumerate(indices):
            if (
                position // self.config.micro_batch_size
            ) % self.dp_world_size != self.dp_rank:
                continue
            index = int(index)
            identity = self.source_id(index)
            rows = encode_pair(
                self.template,
                self.dataset[index],
                identity,
                max_length=self.max_length,
                token_budget=self.token_budget,
            )
            pairs.append((identity, rows))
        groups = pack_pairs(
            pairs,
            token_budget=self.token_budget,
            max_pairs=self.config.micro_batch_size,
            packing=self.config.packing,
        )
        batches = [self.collate(group) for group in groups]
        self.cursor += len(indices)
        return batches, len(indices)

    def collate(self, pairs):
        rows = [row for _, pair in pairs for row in pair]
        batch = ArrowDataLoader._collate(self, rows)
        ids = torch.full((self.token_budget,), -1, dtype=torch.long)
        offset = 0
        counts = []
        for sequence, row in enumerate(rows):
            size = row["input_ids"].numel()
            ids[offset : offset + size] = sequence
            counts.append(int((row["labels"] != -100).sum()))
            offset += size
        # A dummy pair keeps all head/decoder/FSDP collectives and contributes zero.
        valid = torch.ones(max(1, len(pairs)), dtype=torch.bool)
        if not pairs:
            valid.zero_()
            counts = [0, 0]
        return PreferenceMicrobatch(
            input=batch.input,
            labels=batch.labels,
            positions=batch.positions,
            padding_mask=batch.padding_mask,
            model_kwargs=batch.model_kwargs,
            num_valid_tokens=batch.num_valid_tokens,
            document_lengths=batch.document_lengths,
            sequence_ids=ids,
            pair_ids=tuple(identity for identity, _ in pairs),
            pair_valid=valid,
            completion_counts=torch.tensor(counts, dtype=torch.long).reshape(-1, 2),
        )

    def __iter__(self):
        return self

    def __next__(self):
        raise TypeError(
            "PreferenceDataLoader must be consumed with next_step() to preserve pair ownership"
        )

    def state_dict(self):
        return dict(
            version=PLAN_VERSION,
            epoch=self.epoch,
            cursor=self.cursor,
            fingerprint=self.fingerprint,
            pending=[],
        )

    def load_state_dict(self, state):
        if (
            state["version"] != PLAN_VERSION
            or state["fingerprint"] != self.fingerprint
            or state["pending"]
        ):
            raise ValueError(
                "DPO data/template/packing configuration differs from checkpoint"
            )
        if not 0 <= state["cursor"] <= len(self.dataset) or state["epoch"] < 0:
            raise ValueError("Invalid DPO data cursor")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self._set_order()
