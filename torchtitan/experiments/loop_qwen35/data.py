# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Stateful raw or prepared Arrow SFT input for the native Titan trainer."""

import copy
import hashlib
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_from_disk, load_dataset
from pathlib import Path

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import (
    DatasetBuildContext,
    TokenizedTrainingMicrobatch,
)

from torchtitan.experiments.loop_qwen35.length_cache import (
    build_template,
    encoding_fingerprint,
    load_cache,
)
from torchtitan.experiments.loop_qwen35.packing import (
    balanced_plan,
    PLAN_VERSION,
    read_lengths,
)
from torchtitan.experiments.loop_qwen35.packing_vision import (
    geometry_hash,
    image_grid,
    load_vision_cache,
    VISION_PLAN_VERSION,
    visual_row_costs,
)
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class ArrowMicrobatch(TokenizedTrainingMicrobatch):
    document_lengths: tuple[int, ...] = ()


def _data_phase(loader, name):
    # The collator is shared with preference loaders, which do not inherit the
    # SFT loader. Keep optional timing independent of that class hierarchy.
    context = getattr(loader, "_pack_context", None)
    timer = getattr(context, "timer", getattr(loader, "step_timing", None))
    return timer.phase(name) if timer is not None else nullcontext()


def load_dataset_sources(paths):
    """Read local JSONL files or Hugging Face save_to_disk directories."""
    return [
        load_from_disk(path)
        if Path(path).is_dir()
        else load_dataset("json", data_files=path, split="train")
        for path in paths
    ]


class ArrowDataLoader(BaseDataLoader):
    """Encode raw conversations, preserving images and assistant-only labels.

    TP/CP peers share the same DP input. Greedy mode keeps a per-rank
    row cursor. Balanced mode assigns adjacent-cost packs from a shared global
    plan to DP ranks. Online encoding must match its offline lengths exactly.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        paths: list[str] = field(default_factory=list)
        model_path: str = ""
        packing: bool = True
        padding_free: bool = True
        seed: int = 42
        packing_strategy: str = "greedy"
        packing_dp_balance: bool = True
        packing_buffer_size: int = 8192
        packing_length_cache: str = ""
        packing_prefetch_packs: int = 0
        packing_prefetch_mode: str = "row"
        packing_prefetch_max_mib: int = 1024
        packing_vision_balance: bool = False
        packing_vision_weights: dict = field(default_factory=dict)
        grad_accumulation: int = 1

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

        if not config.padding_free:
            raise ValueError(
                "Titan Loop requires padding_free=true (flat token layout)"
            )
        if config.packing_strategy not in ("greedy", "balanced"):
            raise ValueError("packing_strategy must be greedy or balanced")
        if config.packing_strategy == "balanced" and not config.packing:
            raise ValueError("Balanced packing requires packing=true")
        if config.packing_length_cache and config.packing_strategy != "balanced":
            raise ValueError("Packing length cache requires balanced packing")
        if config.packing_vision_balance and config.packing_strategy != "balanced":
            raise ValueError("Visual balance requires balanced packing")
        if config.grad_accumulation < 1:
            raise ValueError("grad_accumulation must be positive")
        if config.packing_buffer_size < 1:
            raise ValueError("packing_buffer_size must be positive")
        if config.packing_prefetch_packs < 0:
            raise ValueError("packing_prefetch_packs must be nonnegative")
        if config.packing_prefetch_packs and config.packing_strategy != "balanced":
            raise ValueError("CPU encoding prefetch requires balanced packing")
        if config.packing_prefetch_mode not in ("row", "pack"):
            raise ValueError("packing_prefetch_mode must be row or pack")
        if config.packing_prefetch_max_mib < 1:
            raise ValueError("packing_prefetch_max_mib must be positive")
        if config.packing_prefetch_mode == "pack" and not config.packing_prefetch_packs:
            raise ValueError("Final pack prefetch requires a positive prefetch depth")
        self.config = config
        self._row_prefetch = None
        datasets = load_dataset_sources(config.paths)
        self.dataset_offsets = np.cumsum([0] + [len(ds) for ds in datasets]).tolist()
        self.dataset = concatenate_datasets(datasets)
        if not {"messages", "input_ids", "raw_sample_json"}.intersection(
            self.dataset.column_names
        ):
            raise ValueError("Expected messages or prepared SFT input_ids/labels")
        if len(self.dataset) < dp_world_size:
            raise ValueError("Dataset has fewer rows than DP ranks")
        self.dp_rank, self.dp_world_size = dp_rank, dp_world_size
        self.token_budget = num_tokens_per_microbatch
        self.max_length = max_context_length
        self._plan_cache = {}
        self.vision_costs = None
        self.encoding_fingerprint = encoding_fingerprint(datasets, config.model_path)
        if config.packing_strategy == "balanced":
            if "lengths" not in self.dataset.column_names:
                raise ValueError(
                    "Balanced packing requires a lengths column; use greedy for raw data without lengths"
                )
            self.row_lengths = read_lengths(self.dataset.data.column("lengths"))
            if config.packing_length_cache:
                ids, lengths = load_cache(
                    config.packing_length_cache,
                    self.encoding_fingerprint,
                    len(self.dataset),
                )
                if len(ids) != len(self.dataset):
                    raise ValueError(
                        "A complete packing length cache is required; finish preprocessing first"
                    )
                self.row_lengths[ids] = lengths
            self.lengths_hash = hashlib.sha256(self.row_lengths.tobytes()).hexdigest()
            if config.packing_vision_balance:
                if not config.packing_length_cache:
                    raise ValueError(
                        "Visual balance requires --packing-length-cache built by "
                        "torchtitan.experiments.loop_qwen35.length_cache_builder; disable with --packing-vision-balance false"
                    )
                ids, _, offsets, grids = load_vision_cache(
                    config.packing_length_cache,
                    self.encoding_fingerprint,
                    len(self.dataset),
                )
                inverse = np.empty(len(ids), dtype=np.int64)
                inverse[ids] = np.arange(len(ids))
                self.image_cache_rows, self.image_offsets, self.image_grids = (
                    inverse,
                    offsets,
                    grids,
                )
                self.image_geometry_hash = geometry_hash(ids, offsets, grids)
                self.vision_costs = np.empty(len(ids), dtype=np.float64)
                self.vision_costs[ids] = visual_row_costs(
                    offsets, grids, config.packing_vision_weights
                )
        self.template = build_template(config.model_path)
        init_logger()
        if config.packing_vision_balance:
            logger.info(
                "Visual packing v6: weights=%s GA=%d images=%d geometry=%s",
                config.packing_vision_weights,
                config.grad_accumulation,
                len(self.image_grids),
                self.image_geometry_hash,
            )
        if config.packing_strategy == "balanced":
            logger.info(
                "Packing DP scheduling: %s",
                "cost-balanced"
                if config.packing_dp_balance
                else "random whole packs (balance disabled)",
            )
        if config.packing_strategy == "balanced" and config.packing_length_cache:
            logger.info(
                "Strict packing: complete encoded-length cache covers %d rows", len(ids)
            )
        context = DatasetBuildContext(
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            num_tokens_per_microbatch=num_tokens_per_microbatch,
            read_options=grain.ReadOptions(),
        )
        self.collator = MultiModalCollator.Config(build_mrope_positions=True).build(
            context=context
        )
        self.epoch = self.cursor = self.skipped = 0
        self._set_order()

    def _set_order(self):
        if self.config.packing_strategy == "balanced":
            self.order = self._plan(self.epoch)
            return
        order = np.random.default_rng(self.config.seed + self.epoch).permutation(
            len(self.dataset)
        )
        self.order = order[self.dp_rank :: self.dp_world_size]

    def _plan(self, epoch):
        if epoch not in self._plan_cache:
            self._plan_cache[epoch] = balanced_plan(
                self.row_lengths,
                token_budget=self.token_budget,
                max_length=self.max_length,
                dp_world_size=self.dp_world_size,
                seed=self.config.seed,
                epoch=epoch,
                buffer_size=self.config.packing_buffer_size,
                vision_costs=getattr(self, "vision_costs", None),
                grad_accumulation=self.config.grad_accumulation,
                dp_balance=self.config.packing_dp_balance,
            )
        return self._plan_cache[epoch]

    def _finish_balanced_epoch(self):
        if self.cursor == len(self.order):
            self.epoch += 1
            self.cursor = 0
            self._set_order()
            self._plan_cache = {self.epoch: self.order}

    def _advance(self):
        self.cursor += 1
        if self.cursor == len(self.order):
            self.epoch += 1
            self.cursor = 0
            self._set_order()

    def __iter__(self):
        return self

    def __next__(self):
        if self.config.packing_strategy == "balanced":
            if self.config.packing_prefetch_mode == "pack":
                self._schedule_packs()
                result = self._next_balanced()
                self._schedule_packs()
                return result
            if self.config.packing_prefetch_packs and self._row_prefetch is None:
                from torchtitan.experiments.loop_qwen35.row_prefetch import (
                    RowEncodingPrefetch,
                )

                self._row_prefetch = RowEncodingPrefetch(self._encode_source_row)
                self._schedule_encoding()
            result = self._next_balanced()
            self._schedule_encoding()
            return result
        rows, used, attempts = [], 0, 0
        invalid_epoch = self.epoch
        while used < self.token_budget:
            index = int(self.order[self.cursor])
            sample = copy.deepcopy(self.dataset[index])
            # Rejected answers are irrelevant to SFT, but retain messages,
            # image bytes/paths, tools and all other template input fields.
            sample.pop("rejected_messages", None)
            encoded = self.template.encode(sample)
            tokens = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
            labels = torch.as_tensor(encoded["labels"], dtype=torch.long)
            if (
                len(tokens) > min(self.max_length, self.token_budget)
                or not (labels[1:] != -100).any()
            ):
                self.skipped += 1
                # A reshuffle can repeat rows across an epoch boundary.
                attempts = attempts + 1 if invalid_epoch == self.epoch else 1
                invalid_epoch = self.epoch
                logger.warning(
                    "Skipping row %d: tokens=%d, max=%d or no SFT targets",
                    index,
                    len(tokens),
                    self.max_length,
                )
                self._advance()
                if attempts >= len(self.order):
                    raise ValueError(
                        "No usable SFT sample in a complete DP-local dataset pass"
                    )
                continue
            if rows and used + len(tokens) > self.token_budget:
                break
            shifted = torch.cat([labels[1:], torch.full((1,), -100, dtype=torch.long)])
            row = {
                "input_ids": tokens,
                "labels": shifted,
                "positions": torch.arange(len(tokens)),
                **{
                    key: encoded[key]
                    for key in ("pixel_values", "image_grid_thw")
                    if key in encoded
                },
            }
            if encoded.get("pixel_values_videos") is not None:
                raise NotImplementedError(
                    "This Arrow recipe supports text/images, not video timestamp packing"
                )
            rows.append(row)
            used += len(tokens)
            self._advance()
            if not self.config.packing:
                break
        return self._collate(rows)

    def _encode_source_row(self, index):
        sample = copy.deepcopy(self.dataset[index])
        sample.pop("rejected_messages", None)
        return self.template.encode(sample)

    def _schedule_encoding(self):
        if self._row_prefetch is None:
            return
        stop = min(
            len(self.order),
            self.cursor + self.config.packing_prefetch_packs * self.dp_world_size,
        )
        indices = (
            index
            for cursor in range(self.cursor + self.dp_rank, stop, self.dp_world_size)
            for index in self.order[cursor]
        )
        self._row_prefetch.schedule(indices)

    def _schedule_packs(self):
        if getattr(self, "_pack_prefetch", None) is None:
            from threading import local

            from torchtitan.experiments.loop_qwen35.pack_prefetch import PackPrefetch

            self._pack_context = local()
            self._pack_prefetch = PackPrefetch(
                self._prepare_pack,
                depth=self.config.packing_prefetch_packs,
                max_bytes=self.config.packing_prefetch_max_mib * 2**20,
            )
        epoch, cursor = self.epoch, self.cursor
        entries = []
        for _ in range(self.config.packing_prefetch_packs):
            plan = self._plan(epoch)
            entries.append(((epoch, cursor), plan[cursor + self.dp_rank]))
            cursor += self.dp_world_size
            if cursor == len(plan):
                epoch, cursor = epoch + 1, 0
        self._pack_prefetch.schedule(entries)

    def _prepare_pack(self, indices):
        # This timer is private to the CPU producer, never collected across ranks.
        from torchtitan.experiments.loop_qwen35.step_timing import StepTiming

        timer = StepTiming(
            "cpu", enabled=getattr(getattr(self, "step_timing", None), "enabled", False)
        )
        self._pack_context.timer = timer
        timer.begin_step()
        try:
            with timer.phase("data_local"):
                result = self._build_balanced_pack(indices)
            return result, dict(timer.host)
        finally:
            timer.end_step()
            del self._pack_context.timer

    def close(self):
        if getattr(self, "_pack_prefetch", None) is not None:
            self._pack_prefetch.close()
            self._pack_prefetch = None
        if self._row_prefetch is not None:
            self._row_prefetch.close()
            self._row_prefetch = None

    def _encode_balanced_row(self, index):
        with _data_phase(self, "encode_wait"):
            encoded = (
                self._row_prefetch.get(index)
                if self._row_prefetch is not None
                else self._encode_source_row(index)
            )
        tokens = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        labels = torch.as_tensor(encoded["labels"], dtype=torch.long)
        expected = int(self.row_lengths[index])
        if tokens.ndim != 1 or tokens.numel() != expected:
            raise ValueError(
                f"row={index}, expected_tokens={expected}, actual_tokens={tokens.numel()}; "
                "offline and training encoder/preprocessing must match"
            )
        if labels.shape != tokens.shape or not (labels[1:] != -100).any():
            raise ValueError(
                f"row={index}: invalid labels or no shifted SFT targets; validate offline"
            )
        if encoded.get("pixel_values_videos") is not None:
            raise NotImplementedError(
                "This Arrow recipe supports text/images, not video timestamp packing"
            )
        if self.config.packing_vision_balance:
            cache_row = self.image_cache_rows[index]
            expected_grid = self.image_grids[
                self.image_offsets[cache_row] : self.image_offsets[cache_row + 1]
            ].tolist()
            if image_grid(encoded) != expected_grid:
                raise ValueError(
                    f"row={index}: image_grid_thw differs from offline geometry; "
                    "offline and training image preprocessing must match"
                )
        shifted = torch.cat([labels[1:], torch.full((1,), -100, dtype=torch.long)])
        row = {
            "input_ids": tokens,
            "labels": shifted,
            "positions": torch.arange(len(tokens)),
            **{
                key: encoded[key]
                for key in ("pixel_values", "image_grid_thw")
                if key in encoded
            },
        }
        return row

    def _raise_encoding_error(self, error):
        # All consumers reach the same collective before model execution. Catch
        # worker/encoder/collator failures too: a single-rank raise would strand peers.
        if not self._global_max(int(error is not None)):
            return
        errors = [error]
        if dist.is_initialized():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        details = next(
            (value for value in errors if value), "Encoding failed on another rank"
        )
        raise ValueError(
            f"Strict packing failed: {details}; data cursor was not advanced"
        )

    def _build_balanced_pack(self, indices):
        rows = []
        for index in indices:
            try:
                rows.append(self._encode_balanced_row(index))
            except Exception as exc:
                raise ValueError(f"row={index}: {type(exc).__name__}: {exc}") from exc
        used = sum(row["input_ids"].numel() for row in rows)
        if used > self.token_budget:
            raise ValueError(
                f"Planned pack has {used} tokens > budget {self.token_budget}"
            )
        return self._collate(rows)

    def _next_balanced(self):
        result, error = None, None
        timer = getattr(self, "step_timing", None)
        try:
            with timer.phase("data_local") if timer is not None else nullcontext():
                if self.config.packing_prefetch_mode == "pack":
                    with _data_phase(self, "pack_wait"):
                        result, worker_times = self._pack_prefetch.get(
                            (self.epoch, self.cursor)
                        )
                    if timer is not None and timer.active:
                        for phase in (
                            "data_local",
                            "encode_wait",
                            "collate_text",
                            "collate_images",
                            "collate_mrope",
                        ):
                            timer.host["prefetch_" + phase] += worker_times[phase]
                else:
                    result = self._build_balanced_pack(
                        self.order[self.cursor + self.dp_rank]
                    )
        except Exception as exc:
            error = (
                f"dp_rank={self.dp_rank}, epoch={self.epoch}, cursor={self.cursor}, "
                f"encoder={self.encoding_fingerprint.get('sha256', 'unknown')}: "
                f"{type(exc).__name__}: {exc}"
            )[:4096]
        with timer.phase("data_sync") if timer is not None else nullcontext():
            self._raise_encoding_error(error)
        self.cursor += self.dp_world_size
        with _data_phase(self, "plan_epoch"):
            self._finish_balanced_epoch()
        return result

    @staticmethod
    def _collective_device():
        return (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )

    def _global_max(self, count):
        if dist.is_initialized():
            value = torch.tensor(
                count, dtype=torch.int64, device=self._collective_device()
            )
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            count = int(value.item())
        return count

    def _collate(self, rows):
        used = sum(row["input_ids"].numel() for row in rows)
        if rows:
            with _data_phase(self, "collate_text"):
                tokens, labels, positions, padding = self.collator.collate_text(rows)
        else:
            # The final epoch tail still enters every model collective.
            # Sum-reduced CE contributes zero on these slots; normalization uses
            # the global supervised-token count of the real samples.
            tokens = torch.full(
                (self.token_budget,), self.collator.tokenizer.pad_id, dtype=torch.long
            )
            labels = torch.full_like(tokens, -100)
            positions = torch.arange(self.token_budget) % self.max_length
            padding = torch.ones(self.token_budget, dtype=torch.bool)
        images = [r["pixel_values"] for r in rows if "pixel_values" in r]
        grids = [r["image_grid_thw"] for r in rows if "image_grid_thw" in r]
        model_kwargs = {}
        if images:
            with _data_phase(self, "collate_images"):
                grid = torch.cat(grids)
                pixels = torch.cat(images)
            with _data_phase(self, "collate_mrope"):
                mrope = self.collator._build_mrope_positions(
                    tokens,
                    grid,
                    None,
                    positions,
                    image_token_id=self.collator.tokenizer.image_id,
                    video_token_id=self.collator.tokenizer.video_id,
                )
            model_kwargs = {
                "pixel_values": pixels,
                "grid_thw": grid,
                "special_tokens": {
                    "image_id": self.collator.tokenizer.image_id,
                    "video_id": self.collator.tokenizer.video_id,
                },
                "mrope_positions": mrope,
            }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Arrow microbatch: rows=%d tokens=%d padding=%d supervised=%d images=%d",
                len(rows),
                used,
                int(padding.sum()),
                int((labels != -100).sum()),
                sum(len(g) for g in grids),
            )
        return ArrowMicrobatch(
            document_lengths=tuple(row["input_ids"].numel() for row in rows),
            input=tokens,
            labels=labels,
            positions=positions,
            padding_mask=padding,
            num_valid_tokens=int((labels != -100).sum()),
            model_kwargs=model_kwargs,
        )

    def state_dict(self):
        state = {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "skipped": self.skipped,
            "paths": self.config.paths,
            "model_path": self.config.model_path,
            "rows": len(self.dataset),
            "seed": self.config.seed,
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "token_budget": self.token_budget,
            "max_length": self.max_length,
            "packing": self.config.packing,
            "encoding_fingerprint": self.encoding_fingerprint,
        }
        state["packing_strategy"] = self.config.packing_strategy
        if self.config.packing_strategy == "balanced":
            state.update(
                packing_dp_balance=self.config.packing_dp_balance,
                packing_buffer_size=self.config.packing_buffer_size,
                packing_plan_version=VISION_PLAN_VERSION
                if self.config.packing_vision_balance
                else PLAN_VERSION,
                lengths_hash=self.lengths_hash,
            )
            if self.config.packing_vision_balance:
                state.update(
                    image_geometry_hash=self.image_geometry_hash,
                    packing_vision_weights=self.config.packing_vision_weights,
                    grad_accumulation=self.config.grad_accumulation,
                )
        return state

    def load_state_dict(self, state):
        state = dict(state)
        current = self.state_dict()
        if (
            self.config.packing_strategy == "balanced"
            and state.get("packing_plan_version") != current["packing_plan_version"]
        ):
            raise ValueError(
                "Cannot resume changed packing_plan_version: keep the original plan/cost settings "
                "or start a new data stage; old cursors must not be discarded"
            )
        for key in current.keys() - {"epoch", "cursor", "skipped"}:
            if state.get(key) != current[key]:
                raise ValueError(
                    f"Cannot resume dataloader with changed {key}; keep the original data plan "
                    "or start a new training stage/output directory"
                )
        epoch, cursor, skipped = (state[k] for k in ("epoch", "cursor", "skipped"))
        if any(not isinstance(n, int) or n < 0 for n in (epoch, cursor, skipped)):
            raise ValueError("Invalid dataloader cursor/epoch/skipped state")
        if self.config.packing_strategy == "balanced":
            plan = self._plan(epoch)
            if cursor >= len(plan) or cursor % self.dp_world_size:
                raise ValueError("Invalid balanced dataloader cursor")
        else:
            if cursor >= len(self.order):
                raise ValueError("Invalid greedy dataloader cursor")
        self.close()
        self.epoch, self.cursor, self.skipped = epoch, cursor, skipped
        self._set_order()
