# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import pytest
import torch
from types import SimpleNamespace

from torchtitan.experiments.SenseNova_Loop_9B_Preview.dpo.data import (
    PreferenceDataLoader,
    encode_pair,
)
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator


def pair():
    return dict(
        messages=[
            dict(role="user", content="Hello?"),
            dict(role="assistant", content="history."),
            dict(role="user", content="question?"),
            dict(role="assistant", content="good answer"),
        ],
        rejected_response="bad answer",
    )


def encode(template, row):
    return encode_pair(template, row, "dataset:7", max_length=512, token_budget=1024)


def test_public_formats_and_last_round_mask(template):
    row = pair()
    alternative = copy.deepcopy(row)
    alternative["rejected_messages"] = alternative["messages"][:-1] + [
        dict(role="assistant", content="bad answer")
    ]
    del alternative["rejected_response"]
    first, second = encode(template, row), encode(template, alternative)
    for a, b in zip(first, second):
        torch.testing.assert_close(a["input_ids"], b["input_ids"])
        torch.testing.assert_close(a["labels"], b["labels"])
        assert a["labels"][-1] == -100
        # The earlier assistant is part of the common, unscored prompt.
        encoded = template.encode(copy.deepcopy(row))
        expected = torch.tensor(
            encoded["chosen_labels" if a is first[0] else "rejected_labels"][1:]
            + [-100]
        )
        torch.testing.assert_close(a["labels"], expected)
        first_target = int(torch.where(a["labels"] != -100)[0][0])
        assert first_target > len(a["labels"]) // 2


@pytest.mark.parametrize(
    "kind",
    ["missing", "identical", "prompt", "weight", "empty", "video", "images"],
)
def test_bad_pairs_report_source(template, kind):
    row = pair()
    if kind == "missing":
        del row["rejected_response"]
    elif kind == "identical":
        row["rejected_response"] = row["messages"][-1]["content"]
    elif kind == "prompt":
        row["rejected_messages"] = [
            dict(role="user", content="different"),
            dict(role="assistant", content="bad"),
        ]
        del row["rejected_response"]
    elif kind == "weight":
        row["messages"][-1]["loss_scale"] = 0.5
    elif kind == "empty":
        row["messages"][-1]["loss"] = False
    elif kind == "video":
        row["videos"] = ["a.mp4"]
    elif kind == "images":
        row.update(images=["a.png"], rejected_images=["b.png"])
    with pytest.raises(ValueError, match="dataset:7"):
        encode(template, row)


def test_pair_length_is_never_truncated(template):
    with pytest.raises(ValueError, match="pair lengths"):
        encode_pair(template, pair(), "row:1", max_length=4, token_budget=8)


def loader(template, *, rank, world, packing, rows=5):
    result = PreferenceDataLoader.__new__(PreferenceDataLoader)
    result.config = PreferenceDataLoader.Config(
        paths=["fixture"],
        packing=packing,
        seed=42,
        micro_batch_size=2,
        global_batch_size=4 * world,
    )
    result.template = template
    result.dp_rank, result.dp_world_size = rank, world
    result.dataset = [pair() for _ in range(rows)]
    result.dataset_offsets = [0, rows]
    result.max_length, result.token_budget = 512, 1024
    result.epoch = result.cursor = 0
    result.fingerprint = "fixture-contract"
    result.collator = MultiModalCollator.Config().build(
        context=SimpleNamespace(
            tokenizer=SimpleNamespace(pad_id=0),
            max_context_length=512,
            num_tokens_per_microbatch=1024,
        )
    )
    result._set_order()
    return result


@pytest.mark.parametrize("packing", [False, True])
def test_step_ownership_tail_dummy_and_resume(template, packing):
    loaders = [loader(template, rank=i, world=4, packing=packing) for i in range(4)]
    steps = [item.next_step() for item in loaders]
    identities = [
        identity
        for batches, count in steps
        for batch in batches
        for identity in batch.pair_ids
    ]
    assert len(identities) == len(set(identities)) == 5
    assert all(count == 5 for _, count in steps)
    assert not steps[-1][0]
    dummy = loaders[-1].collate([])
    assert dummy.num_valid_pairs == dummy.num_valid_tokens == 0
    assert (dummy.sequence_ids == -1).all()
    for batch in steps[0][0]:
        start = 0
        for length in batch.document_lengths:
            assert (
                batch.positions[start] == 0 and batch.labels[start + length - 1] == -100
            )
            start += length
    state = loaders[0].state_dict()
    expected = loaders[0].next_step()
    loaders[0].load_state_dict(state)
    actual = loaders[0].next_step()
    assert [b.pair_ids for b in actual[0]] == [b.pair_ids for b in expected[0]]
    bad = dict(state, fingerprint="changed")
    with pytest.raises(ValueError, match="differs"):
        loaders[0].load_state_dict(bad)


def trajectory():
    return dict(
        messages=[
            dict(role="user", content="Q1"),
            dict(role="assistant", content="A1"),
            dict(role="user", content="Q2"),
            dict(role="assistant", content="A2"),
        ],
        rejected_messages=[
            dict(role="user", content="Q1"),
            dict(role="assistant", content="B1"),
            dict(role="user", content="Q3"),
            dict(role="assistant", content="B2"),
            dict(role="user", content="Q4"),
            dict(role="assistant", content="B3"),
        ],
    )


def test_default_scores_all_assistant_turns_and_allows_divergence(template):
    template._loss_scale = "default"
    rows = encode(template, trajectory())
    for row, answers in zip(rows, [("A1", "A2"), ("B1", "B2", "B3")]):
        targets = row["labels"][row["labels"] != -100]
        # Independently specify the expected supervised text, including EOS.
        expected = template.tokenizer.encode(
            "".join(answer + "<|im_end|>\n" for answer in answers),
            add_special_tokens=False,
        )
        assert targets.tolist() == expected
    template._loss_scale = "last_round"
    with pytest.raises(ValueError, match="prompts differ"):
        encode(template, trajectory())


@pytest.mark.parametrize(
    "change",
    ["initial_user", "system", "tools", "kwargs", "no_answers", "unscored_history"],
)
def test_default_still_validates_conditioning_context(template, change):
    template._loss_scale = "default"
    row = trajectory()
    if change == "initial_user":
        row["rejected_messages"][0]["content"] = "another task"
    elif change == "system":
        row["rejected_messages"].insert(
            0, dict(role="system", content="different system")
        )
    elif change == "tools":
        row["tools"] = [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}},
            }
        ]
        row["rejected_tools"] = []
    elif change == "kwargs":
        row["chat_template_kwargs"] = {"enable_thinking": False}
        row["rejected_chat_template_kwargs"] = {"enable_thinking": True}
    elif change == "no_answers":
        for message in row["messages"]:
            message["loss"] = False
    else:
        # A divergent unscored prefix is not a shared conditioning context.
        row["messages"][1]["loss"] = False
        row["rejected_messages"][1]["loss"] = False
    with pytest.raises(ValueError, match="dataset:7"):
        encode(template, row)


@pytest.mark.parametrize("divergent", [False, True])
def test_trajectory_loss_and_gradient_against_explicit_causal_reference(
    template, divergent
):
    import torch.nn.functional as F

    from torchtitan.experiments.SenseNova_Loop_9B_Preview.dpo.loss import dpo_loss
    from torchtitan.components.sequence_logps import (
        sequence_logps,
        weighted_sequence_backward,
    )

    template._loss_scale = "default"
    source = trajectory() if divergent else pair()
    rows = encode(template, source)
    torch.manual_seed(7)
    embedding = torch.nn.Embedding(len(template.tokenizer), 7, dtype=torch.float64)
    head = torch.nn.Linear(7, len(template.tokenizer), bias=False, dtype=torch.float64)
    other_embedding, other_head = copy.deepcopy((embedding, head))
    # A causal model with a separate state per document; no cross-pair leakage.
    labels = torch.cat([row["labels"] for row in rows])
    ids = torch.cat(
        [torch.full_like(row["input_ids"], i) for i, row in enumerate(rows)]
    )
    reference = torch.tensor([[-11.0, -12.0]], dtype=torch.float64)
    manual_scores = []
    for row in rows:
        logits = head(embedding(row["input_ids"]).cumsum(0).tanh())
        manual_scores.append(
            -F.cross_entropy(logits, row["labels"], reduction="sum", ignore_index=-100)
        )
    manual_scores = torch.stack(manual_scores).reshape(1, 2)
    manual_loss = -F.logsigmoid(
        0.1 * ((manual_scores[0, 0] - manual_scores[0, 1]) - 1.0)
    )
    manual_loss.backward()
    other_hidden = torch.cat(
        [other_embedding(row["input_ids"]).cumsum(0).tanh() for row in rows]
    )
    padding = (-labels.numel()) % 4
    other_hidden = F.pad(other_hidden, (0, 0, 0, padding))
    labels = F.pad(labels, (0, padding), value=-100)
    ids = F.pad(ids, (0, padding), value=-1)
    with torch.no_grad():
        scores = sequence_logps(
            other_hidden.detach(), other_head, labels, ids, 2, num_chunks=4
        ).reshape(1, 2)
        loss, coefficients, _ = dpo_loss(scores, reference, torch.tensor([True]))
    weighted_sequence_backward(
        other_hidden, other_head, labels, ids, coefficients.flatten(), num_chunks=4
    )
    torch.testing.assert_close(scores, manual_scores, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(loss, manual_loss, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        other_embedding.weight.grad, embedding.weight.grad, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        other_head.weight.grad, head.weight.grad, atol=1e-10, rtol=1e-10
    )
    if not divergent:
        # Under causal shared history, scoring earlier shared answers cancels.
        # Hold tokens fixed: Qwen last_round also removes historical thinking.
        shared = next(
            i
            for i, (a, b) in enumerate(zip(rows[0]["input_ids"], rows[1]["input_ids"]))
            if a != b
        )
        last_scores = []
        for row in rows:
            labels = row["labels"].clone()
            labels[: shared - 1] = -100
            logits = head(embedding(row["input_ids"]).cumsum(0).tanh())
            last_scores.append(
                -F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
            )
        last_loss = -F.logsigmoid(0.1 * ((last_scores[0] - last_scores[1]) - 1.0))
        last_grad = torch.autograd.grad(last_loss, (embedding.weight, head.weight))
        torch.testing.assert_close(last_loss, manual_loss, atol=1e-10, rtol=1e-10)
        for grad, expected in zip(last_grad, (embedding.weight.grad, head.weight.grad)):
            torch.testing.assert_close(grad, expected, atol=1e-10, rtol=1e-10)


def test_original_pair_is_not_mutated(template):
    source = pair()
    before = copy.deepcopy(source)
    encode(template, source)
    assert source == before


@pytest.mark.parametrize("warmup", [0, 2, 4])
def test_sft_scheduler_unchanged(warmup):
    import math
    from torchtitan.experiments.SenseNova_Loop_9B_Preview.mhc.schedule import factor

    for step in range(41):
        total = 40
        expected = (
            (step + 1) / warmup
            if step < warmup
            else (
                1
                + math.cos(
                    math.pi
                    * min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
                )
            )
            / 2
        )
        assert factor(step, total, warmup, 0) == expected
        assert factor(step, total, warmup, 0, decay_ratio=0) == (
            (step + 1) / warmup if step < warmup else 1
        )


@pytest.mark.parametrize("storage", ["jsonl", "arrow"])
def test_raw_dataset_loads_without_prepare(template, tmp_path, storage):
    import json
    from datasets import Dataset

    source = tmp_path / storage
    samples = [pair() for _ in range(5)]
    if storage == "arrow":
        Dataset.from_list(samples).save_to_disk(str(source))
    else:
        source.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
    config = PreferenceDataLoader.Config(
        paths=[str(source)],
        model_path=str(tmp_path),
        global_batch_size=4,
        micro_batch_size=1,
    )
    item = config.build(
        dp_world_size=2,
        dp_rank=0,
        tokenizer=SimpleNamespace(pad_id=0),
        max_context_length=512,
        num_tokens_per_microbatch=1024,
    )
    batches, global_pairs = item.next_step()
    assert global_pairs == 4 and sum(b.num_valid_pairs for b in batches) == 2
    assert all(b.num_valid_tokens > 0 for b in batches)
    state = item.state_dict()
    expected = item.next_step()
    item.load_state_dict(state)
    actual = item.next_step()
    assert [b.pair_ids for b in expected[0]] == [b.pair_ids for b in actual[0]]
    item.close()
