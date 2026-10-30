# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Packing coverage, rank balance, resume and unchunked-loss semantics on CPU."""

import copy
import hashlib
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch


import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchtitan.components.loss import (  # noqa: E402
    ChunkedLossWrapper,
    CrossEntropyLoss,
)

from torchtitan.experiments.SenseNova_Loop_9B_Preview import train  # noqa: E402
from torchtitan.experiments.SenseNova_Loop_9B_Preview.data import (
    ArrowDataLoader,
)  # noqa: E402
from torchtitan.experiments.SenseNova_Loop_9B_Preview.packing import (  # noqa: E402
    attention_cost,
    balanced_plan,
    read_lengths,
)
from torchtitan.hf_datasets.multimodal.mm_collator import (  # noqa: E402
    MultiModalCollator,
)

ROOT = Path(__file__).resolve().parents[3]


def loader(
    actual,
    *,
    estimated=None,
    rank=0,
    dp=1,
    budget=128,
    strategy="balanced",
    images=False,
):
    obj = ArrowDataLoader.__new__(ArrowDataLoader)
    obj.config = ArrowDataLoader.Config(
        paths=["fixture"],
        model_path="fixture",
        seed=42,
        packing_strategy=strategy,
        packing_buffer_size=128,
    )
    obj.dataset = [
        {"messages": [], "row_id": i, "length": n} for i, n in enumerate(actual)
    ]
    obj.dataset_offsets = [0, len(actual)]
    obj.dp_rank, obj.dp_world_size = rank, dp
    obj.token_budget = obj.max_length = budget
    obj.row_lengths = np.asarray(estimated or actual, dtype=np.int64)
    obj.lengths_hash = hashlib.sha256(obj.row_lengths.tobytes()).hexdigest()
    obj._plan_cache = {}
    obj._row_prefetch = None
    obj.encoding_fingerprint = {"fixture": True}
    obj.epoch = obj.cursor = obj.skipped = 0

    def encode(row):
        n, ident = row["length"], row["row_id"] + 1
        result = dict(input_ids=[ident] * n, labels=[-100] + [ident] * (n - 1))
        if images:
            result.update(
                pixel_values=torch.tensor([[ident]], dtype=torch.float32),
                image_grid_thw=torch.tensor([[1, 1, ident]]),
            )
        return result

    obj.template = SimpleNamespace(encode=encode)
    obj.collator = MultiModalCollator.__new__(MultiModalCollator)
    obj.collator.tokenizer = SimpleNamespace(pad_id=0, image_id=999, video_id=998)
    obj.collator._num_tokens_per_microbatch = budget
    obj.collator._max_context_length = budget
    obj.collator._build_mrope_positions = lambda tokens, *a, **k: torch.zeros(
        (budget, 3), dtype=torch.long
    )
    obj._set_order()
    return obj


def identities(batch):
    return batch.input[(batch.positions == 0) & ~batch.padding_mask].tolist()


def next_group(loaders):
    barrier = Barrier(len(loaders), timeout=10)
    values = [None] * len(loaders)

    def exchange(rank, value, reduce):
        values[rank] = value
        barrier.wait()
        result = max(values) if reduce else [row for rows in values for row in rows]
        barrier.wait()
        return result

    for rank, dl in enumerate(loaders):
        dl._global_max = lambda value, rank=rank: exchange(rank, value, True)
    with ThreadPoolExecutor(max_workers=len(loaders)) as pool:
        return list(pool.map(next, loaders))


class PackingTest(unittest.TestCase):
    def assert_same_batch(self, left, right):
        for name in ("input", "labels", "positions", "padding_mask"):
            self.assertTrue(
                torch.equal(getattr(left, name), getattr(right, name)), name
            )
        self.assertEqual(left.document_lengths, right.document_lengths)
        self.assertEqual(left.num_valid_tokens, right.num_valid_tokens)
        self.assertEqual(left.model_kwargs.keys(), right.model_kwargs.keys())
        for name, value in left.model_kwargs.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, right.model_kwargs[name]), name)
            else:
                self.assertEqual(value, right.model_kwargs[name])

    def test_encoding_prefetch_preserves_images_cursor_and_resume(self):
        for lookahead in (4, 8):
            with self.subTest(lookahead=lookahead):
                actual = [70, 70, 70, 20, 20, 60, 60]
                baseline = loader(actual, images=True)
                candidate = loader(actual, images=True)
                candidate.config.packing_prefetch_packs = lookahead
                restored = loader(actual, images=True)
                restored.config.packing_prefetch_packs = lookahead
                try:
                    for _ in range(8):
                        self.assert_same_batch(next(baseline), next(candidate))
                        self.assertEqual(baseline.state_dict(), candidate.state_dict())
                    state = copy.deepcopy(candidate.state_dict())
                    # A pending CPU result is speculative; it must not enter the saved cursor.
                    restored.load_state_dict(state)
                    candidate.load_state_dict(state)
                    for _ in range(8):
                        self.assert_same_batch(next(candidate), next(restored))
                        self.assertEqual(candidate.state_dict(), restored.state_dict())
                finally:
                    baseline.close()
                    candidate.close()
                    restored.close()

    def test_encoding_prefetch_keeps_error_collectives_on_consumer_threads(self):
        for lookahead in (4, 8):
            with self.subTest(lookahead=lookahead):
                from threading import current_thread

                actual = [70, 128, 70, 70, 20, 20, 60, 60, 19]
                baseline = [loader(actual, rank=r, dp=2) for r in range(2)]
                candidate = [loader(actual, rank=r, dp=2) for r in range(2)]
                for dl in candidate:
                    dl.config.packing_prefetch_packs = lookahead
                    encode = dl.template.encode

                    def checked_encode(row, encode=encode):
                        self.assertTrue(
                            current_thread().name.startswith("arrow-encode")
                        )
                        return encode(row)

                    dl.template.encode = checked_encode
                try:
                    # next_group installs barriers in the consumer threads. Any worker-side
                    # collectives would change their ordering or strand a participant.
                    for _ in range(12):
                        left, right = next_group(baseline), next_group(candidate)
                        for a, b, dl_a, dl_b in zip(left, right, baseline, candidate):
                            self.assert_same_batch(a, b)
                            self.assertEqual(dl_a.state_dict(), dl_b.state_dict())
                finally:
                    for dl in baseline + candidate:
                        dl.close()

    def test_arrow_lengths_support_chunk_offsets_and_scalar_columns(self):
        column = pa.chunked_array(
            [pa.array([[99], [2, 3], [7]]).slice(1), pa.array([[11]])]
        )
        self.assertEqual(read_lengths(column).tolist(), [5, 7, 11])
        self.assertEqual(read_lengths(pa.chunked_array([[5, 7]])).tolist(), [5, 7])
        self.assertEqual(read_lengths(pa.chunked_array([[0, 7]])).tolist(), [0, 7])
        for values in ([[None]], [[[-1]]], [[[3, -1]]]):
            with self.assertRaises(ValueError):
                read_lengths(pa.chunked_array(values))

    def test_plan_coverage_capacity_seed_and_attention_balance(self):
        sizes = np.random.default_rng(5).integers(500, 60000, 8000)
        kwargs = dict(
            token_budget=65536,
            max_length=65536,
            dp_world_size=8,
            seed=42,
            epoch=0,
            buffer_size=8192,
        )
        plan = balanced_plan(sizes, **kwargs)
        self.assertEqual(sorted(i for p in plan for i in p), list(range(len(sizes))))
        self.assertEqual(len(plan) % 8, 0)
        self.assertTrue(all(sum(sizes[i] for i in p) <= 65536 for p in plan))
        self.assertEqual(plan, balanced_plan(sizes, **kwargs))
        self.assertNotEqual(plan, balanced_plan(sizes, **{**kwargs, "epoch": 1}))
        costs = np.array(
            [attention_cost([int(sizes[i]) for i in p], 65536, 65536) for p in plan]
        )
        grouped_ratio = costs.reshape(-1, 8).max(axis=1).mean() / costs.mean()
        random_costs = np.random.default_rng(23).permutation(costs)
        random_ratio = random_costs.reshape(-1, 8).max(axis=1).mean() / costs.mean()
        self.assertLess(grouped_ratio, 1.1)
        self.assertLess(grouped_ratio, random_ratio * 0.8)

    def test_all_ranks_cover_epochs_without_dropping_or_repeating_tail(self):
        loaders = [loader([79, 65, 37, 91, 23, 19, 41], rank=r, dp=4) for r in range(4)]
        first_plan = copy.deepcopy(loaders[0].order)
        for epoch in range(3):
            seen = []
            while loaders[0].epoch == epoch:
                for batch in next_group(loaders):
                    seen += identities(batch)
                    self.assertTrue(torch.all(batch.labels[batch.padding_mask] == -100))
            self.assertEqual(sorted(seen), list(range(1, 8)))
            self.assertTrue(
                all(dl.epoch == epoch + 1 and dl.cursor == 0 for dl in loaders)
            )
        self.assertNotEqual(first_plan, loaders[0].order)

    def test_epoch_tail_padding_has_no_targets_or_document_rows(self):
        loaders = [loader([90] * 7, rank=r, dp=4) for r in range(4)]
        batches = [batch for _ in range(2) for batch in next_group(loaders)]
        self.assertEqual(
            sorted(i for batch in batches for i in identities(batch)), list(range(1, 8))
        )
        empty = [batch for batch in batches if batch.num_valid_tokens == 0]
        self.assertEqual(len(empty), 1)
        self.assertTrue(empty[0].padding_mask.all())
        self.assertTrue((empty[0].labels == -100).all())

    def test_strict_length_mismatch_fails_even_when_pack_fits(self):
        for actual in [50, 70]:
            for depth in [0, 2]:
                dl = loader([actual, 60], estimated=[60, 60], dp=1)
                dl.config.packing_prefetch_packs = depth
                before = copy.deepcopy(dl.state_dict())
                try:
                    with self.assertRaisesRegex(
                        ValueError, f"expected_tokens=60, actual_tokens={actual}"
                    ):
                        next(dl)
                    self.assertEqual(dl.state_dict(), before)
                finally:
                    dl.close()

    def test_tp_peers_labels_positions_and_image_order_match(self):
        left = loader([20, 30, 40, 50], images=True)
        right = loader([20, 30, 40, 50], images=True)
        for _ in range(6):
            a, b = next(left), next(right)
            torch.testing.assert_close(a.input, b.input)
            ids = identities(a)
            if ids:
                self.assertEqual(a.model_kwargs["pixel_values"].flatten().tolist(), ids)
                self.assertEqual(a.model_kwargs["grid_thw"][:, 2].tolist(), ids)
            # Each document's last label is ignored, not shifted into the next document.
            starts = (a.positions == 0).nonzero().flatten().tolist()
            for end in starts[1:] + [a.input.numel()]:
                self.assertEqual(int(a.labels[end - 1]), -100)

    def test_fixed_plan_resume_has_no_runtime_repair_state(self):
        actual = [70, 70, 70, 70, 20, 20, 60, 60]
        ranks = [loader(actual, rank=r, dp=2) for r in range(2)]
        first = next_group(ranks)
        restored = [loader(actual, rank=r, dp=2) for r in range(2)]
        for left, right in zip(ranks, restored):
            state = copy.deepcopy(left.state_dict())
            self.assertNotIn("pending_rows", state)
            self.assertNotIn("length_overrides", state)
            right.load_state_dict(state)
        seen = [ident for batch in first for ident in identities(batch)]
        while ranks[0].epoch == 0:
            for left, right in zip(next_group(ranks), next_group(restored)):
                self.assert_same_batch(left, right)
                seen += identities(left)
        self.assertEqual(sorted(seen), list(range(1, 9)))

    def test_plan_spreads_tail_documents_without_runtime_repair(self):
        sizes = np.array([40] * 6)
        group = balanced_plan(
            sizes,
            token_budget=128,
            max_length=128,
            dp_world_size=4,
            seed=42,
            epoch=0,
            buffer_size=128,
        )
        self.assertEqual(len(group), 4)
        self.assertTrue(all(group))
        self.assertEqual(sorted(i for pack in group for i in pack), list(range(6)))

    def test_resume_rejects_old_plan_or_changed_encoder(self):
        dl = loader([20, 30, 40, 50])
        state = dl.state_dict()
        for key, value in [
            ("packing_plan_version", 2),
            ("packing_plan_version", 3),
            ("packing_plan_version", 4),
            ("encoding_fingerprint", {"fixture": False}),
        ]:
            with self.subTest(key=key):
                changed = {**state, key: value}
                with self.assertRaisesRegex(ValueError, key):
                    dl.load_state_dict(changed)

    def test_exact_lengths_and_multiple_epochs_have_exact_coverage(self):
        actual = np.random.default_rng(77).integers(2, 129, size=80).tolist()
        actual[
            17
        ] = 0  # Explicit offline exclusion; never encoded or silently skipped online.
        ranks = [loader(actual, rank=r, dp=4) for r in range(4)]
        for epoch in range(2):
            seen = []
            while ranks[0].epoch == epoch:
                batches = next_group(ranks)
                self.assertEqual(len({(dl.epoch, dl.cursor) for dl in ranks}), 1)
                if ranks[0].epoch == epoch:
                    self.assertTrue(all(identities(batch) for batch in batches))
                seen.extend(ident for batch in batches for ident in identities(batch))
            self.assertEqual(
                sorted(seen), [i + 1 for i, length in enumerate(actual) if length > 0]
            )

    def test_invalid_metadata_raises_before_iteration(self):
        for lengths, message in [
            ([129, 130], "offline length"),
            ([0, 0], "No usable SFT sample"),
        ]:
            with (
                self.subTest(lengths=lengths),
                self.assertRaisesRegex(ValueError, message),
            ):
                loader(lengths)

    def test_resume_preserves_greedy_and_rejects_strategy_change(self):
        old = loader([20, 30, 40, 50], strategy="greedy")
        next(old)
        state = old.state_dict()
        restored = loader([20, 30, 40, 50], strategy="greedy")
        restored.load_state_dict(state)
        torch.testing.assert_close(next(old).input, next(restored).input)
        with self.assertRaises(ValueError):
            loader([20, 30, 40, 50]).load_state_dict(state)

    def test_image_identity_and_document_lengths_survive_resume(self):
        original = loader([70] * 4, dp=1, images=True)
        next(original)
        restored = loader([70] * 4, dp=1, images=True)
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        for _ in range(3):
            left, right = next(original), next(restored)
            torch.testing.assert_close(left.input, right.input)
            torch.testing.assert_close(left.labels, right.labels)
            self.assertEqual(left.document_lengths, right.document_lengths)
            self.assertEqual(
                left.model_kwargs["pixel_values"].flatten().tolist(), identities(left)
            )
            self.assertEqual(
                sum(left.document_lengths), int((~left.padding_mask).sum())
            )

    def test_full_capacity_rows_are_not_dropped(self):
        dl = loader([128, 70, 128, 70], dp=1)
        batches = []
        while dl.epoch == 0:
            batches.append(next(dl))
        self.assertEqual(
            sorted(i for batch in batches for i in identities(batch)), [1, 2, 3, 4]
        )
        self.assertEqual(dl.skipped, 0)
        self.assertEqual(sum(not batch.padding_mask.any() for batch in batches), 2)

    def test_encoding_and_collator_errors_do_not_advance_cursor(self):
        for stage in ["encode", "collate", "labels"]:
            dl = loader([20, 30])
            state = copy.deepcopy(dl.state_dict())
            if stage == "encode":
                dl.template.encode = lambda row: (_ for _ in ()).throw(
                    OSError("missing image")
                )
            elif stage == "collate":
                dl._collate = lambda rows: (_ for _ in ()).throw(
                    ValueError("bad image grid")
                )
            else:
                dl.template.encode = lambda row: dict(
                    input_ids=[1] * row["length"], labels=[-100] * row["length"]
                )
            with (
                self.subTest(stage=stage),
                self.assertRaisesRegex(ValueError, "Strict packing failed"),
            ):
                next(dl)
            self.assertEqual(state, dl.state_dict())

    def test_corrupt_resume_cursor_is_rejected(self):
        dl = loader([90] * 7, dp=4)
        state = dl.state_dict()
        for cursor in [-1, 1, 999]:
            with (
                self.subTest(cursor=cursor),
                self.assertRaisesRegex(ValueError, "cursor"),
            ):
                dl.load_state_dict({**state, "cursor": cursor})


class LossTest(unittest.TestCase):
    def test_zero_chunks_cli_and_validation(self):
        argv = [
            "train",
            "--model",
            "unused",
            "--cached-dataset",
            "unused",
            "--output-dir",
            "unused",
            "--loss-chunks",
            "0",
            "--packing-strategy",
            "balanced",
            "--tp",
            "1",
            "--cp",
            "4",
            "--decay-ratio",
            "1",
        ]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, WORLD_SIZE="4"):
            args = train.parse_args()
        self.assertEqual(args.loss_chunks, 0)
        with (
            patch.object(sys, "argv", argv + ["--loss-chunks", "-1"]),
            self.assertRaises(SystemExit),
        ):
            train.parse_args()

    def test_full_and_chunked_loss_match_hidden_and_weight_gradients(self):
        torch.manual_seed(17)
        hidden = torch.randn(12, 8)
        labels = torch.tensor([1, 2, -100, 5, 3, 2, 1, -100, 4, 2, 1, 3])
        valid = (labels != -100).sum()
        head = nn.Linear(8, 7, bias=False)
        other = copy.deepcopy(head)
        x, y = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
        ce = CrossEntropyLoss.Config(global_vocab_size=7).build()
        full, _ = ce(head(x), labels, valid)
        full.backward()
        chunked = ChunkedLossWrapper.Config(
            num_chunks=3, loss_fn=CrossEntropyLoss.Config(global_vocab_size=7)
        ).build()
        chunked.set_lm_head(other)
        split, _ = chunked(y, labels, valid)
        split.backward()
        torch.testing.assert_close(full, split)
        torch.testing.assert_close(x.grad, y.grad)
        torch.testing.assert_close(head.weight.grad, other.weight.grad)
        # An epoch-tail padding rank contributes zero to globally normalized CE.
        padded, _ = ce(head(hidden), torch.full_like(labels, -100), valid)
        self.assertEqual(padded.item(), 0.0)


if __name__ == "__main__":
    unittest.main()


def _distributed_strict_worker(rank, directory, port):
    import json
    from datetime import timedelta

    import torch.distributed as dist

    store = dist.TCPStore(
        "127.0.0.1", port, is_master=False, timeout=timedelta(seconds=60)
    )
    dist.init_process_group(
        "gloo", store=store, rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    try:
        actual = [70, 70, 70, 70, 20, 20, 60, 60]
        dl = loader(actual, rank=rank // 2, dp=2, images=True)
        first = next(dl)
        checkpoint = copy.deepcopy(dl.state_dict())
        restored = loader(actual, rank=rank // 2, dp=2, images=True)
        restored.load_state_dict(checkpoint)
        restored.order = copy.deepcopy(dl.order)
        batches = [identities(first)]
        while dl.epoch == 0:
            batch = next(dl)
            replay = next(restored)
            torch.testing.assert_close(batch.input, replay.input)
            torch.testing.assert_close(batch.labels, replay.labels)
            assert dl.state_dict() == restored.state_dict()
            batches.append(identities(batch))
        Path(directory, f"rank{rank}.json").write_text(json.dumps(batches))
        # Every rank, including unaffected DP/CP peers, must receive the same error.
        for kind in ("longer", "shorter", "worker_error", "bad_labels"):
            broken = loader([60, 60], rank=rank // 2, dp=2)
            broken.config.packing_prefetch_packs = 2
            if rank == 0:
                original = broken.template.encode

                def corrupt(row, kind=kind, original=original):
                    if kind == "worker_error":
                        raise OSError("fixture image unreadable")
                    value = original(row)
                    if kind == "bad_labels":
                        value["labels"] = [-100] * 60
                    else:
                        size = 70 if kind == "longer" else 50
                        value = dict(
                            input_ids=[1] * size, labels=[-100] + [1] * (size - 1)
                        )
                    return value

                broken.template.encode = corrupt
            before = copy.deepcopy(broken.state_dict())
            try:
                try:
                    next(broken)
                except ValueError as error:
                    message = str(error)
                    assert "Strict packing failed" in message
                else:
                    raise AssertionError("All ranks must reject malformed inputs")
                gathered = [None] * 4
                dist.all_gather_object(gathered, message)
                assert len(set(gathered)) == 1
                assert broken.state_dict() == before
            finally:
                broken.close()
    finally:
        dist.destroy_process_group()


def test_real_dp2_cp2_strict_resume_and_error_collectives(tmp_path):
    import json

    import torch.distributed as dist
    import torch.multiprocessing as mp

    # Shared filesystems may not support the locks required by FileStore.
    store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    mp.spawn(
        _distributed_strict_worker,
        args=(str(tmp_path), store.port),
        nprocs=4,
        join=True,
    )
    results = [
        json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(4)
    ]
    assert results[0] == results[1]
    assert results[2] == results[3]
    assert len(results[0]) == len(results[2])
    assert all(batch for rank in (0, 2) for batch in results[rank][:-1])
    assert sorted(
        ident for rank in (0, 2) for batch in results[rank] for ident in batch
    ) == list(range(1, 9))
