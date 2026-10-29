# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import pytest
import torch
import torch.distributed as dist
from types import SimpleNamespace

from torchtitan.experiments.loop_qwen35.dpo.checkpoint import (
    DPOCheckpoint,
    digest_state,
    snapshot_state,
)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    dist.init_process_group(
        "gloo", rank=0, world_size=1, init_method=f"file://{tmp_path}/rendezvous"
    )
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", torch.set_rng_state)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = torch.nn.Linear(3, 2)
    model.loop_settings = SimpleNamespace(
        to_dict=lambda: dict(start=0, end=1, counts=[2])
    )
    model.config = SimpleNamespace(mhc=None)
    reference = torch.nn.Linear(3, 2).requires_grad_(False)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    cursor = {"value": 0}
    loader = SimpleNamespace(
        fingerprint="data1",
        state_dict=lambda: dict(cursor),
        load_state_dict=lambda state: cursor.update(state),
    )
    engine = SimpleNamespace(
        device=torch.device("cpu"),
        parallel_dims=SimpleNamespace(tp=1, cp=1, pp=1, dp_replicate=1, dp_shard=1),
        reference=SimpleNamespace(parts=[reference]),
        num_completed_steps=0,
    )
    engine.state_dict = lambda: dict(step=engine.num_completed_steps)
    engine.load_state_dict = lambda state: setattr(
        engine, "num_completed_steps", state["step"]
    )
    engine.initialize_reference = lambda **kwargs: None
    config = DPOCheckpoint.Config(
        interval=1,
        resolved_config=dict(
            dpo=dict(algorithm="sigmoid", beta=0.1),
            base_model={"name": "fixture"},
            loop_settings=model.loop_settings.to_dict(),
        ),
    )
    result = DPOCheckpoint(
        config,
        model_parts=[model],
        optimizers=optimizer,
        lr_schedulers=scheduler,
        dataloader=loader,
        states={"train_state": engine},
        sd_adapter=None,
        base_folder=tmp_path,
    )
    try:
        yield result
    finally:
        dist.destroy_process_group()


def update(checkpoint):
    optimizer, model = checkpoint.optimizers, checkpoint.parts[0]
    optimizer.zero_grad()
    # Exercise random state as well as optimizer state across resume.
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    checkpoint.schedulers.step()
    checkpoint.engine.num_completed_steps += 1
    checkpoint.dataloader.load_state_dict(
        {"value": checkpoint.engine.num_completed_steps}
    )
    return loss.detach()


def test_resume_restores_reference_optimizer_data_and_rng(checkpoint):
    reference_digest = digest_state(snapshot_state(checkpoint.engine.reference.parts))
    for _ in range(2):
        update(checkpoint)
    checkpoint.save(2)
    expected_losses = [update(checkpoint) for _ in range(3)]
    expected_weights = snapshot_state(checkpoint.parts)
    with torch.no_grad():
        for parameter in checkpoint.engine.reference.parts[0].parameters():
            parameter.add_(10)
    assert checkpoint.load()
    assert checkpoint.engine.num_completed_steps == 2
    assert checkpoint.dataloader.state_dict()["value"] == 2
    assert (
        digest_state(snapshot_state(checkpoint.engine.reference.parts))
        == reference_digest
    )
    for expected in expected_losses:
        torch.testing.assert_close(update(checkpoint), expected, atol=0, rtol=0)
    assert digest_state(snapshot_state(checkpoint.parts)) == digest_state(
        expected_weights
    )


@pytest.mark.parametrize("field", ["algorithm", "data", "topology", "base_model"])
def test_contract_mismatch_is_rejected(checkpoint, field):
    checkpoint.save(1)
    path = checkpoint.folder / "step-1/complete.json"
    metadata = json.loads(path.read_text())
    metadata["dpo"]["contract"][field] = "changed"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        checkpoint.load()


@pytest.mark.parametrize("kind", ["content", "missing", "sft"])
def test_missing_or_tampered_reference_is_rejected(checkpoint, kind):
    checkpoint.save(1)
    path = checkpoint.folder / "step-1/complete.json"
    metadata = json.loads(path.read_text())
    if kind == "sft":
        del metadata["dpo"]
        path.write_text(json.dumps(metadata))
    else:
        snapshot = (
            checkpoint.folder.parent
            / metadata["dpo"]["reference"]["path"]
            / "rank-0.pt"
        )
        if kind == "missing":
            snapshot.unlink()
        else:
            state = torch.load(snapshot, weights_only=True)
            state[0]["weight"].add_(1)
            torch.save(state, snapshot)
    with pytest.raises(ValueError, match="DPO|Reference"):
        checkpoint.load()


def test_switching_loss_scale_rejects_resume(checkpoint):
    checkpoint.config.resolved_config["dpo"]["loss_scale"] = "last_round"
    checkpoint.save(1)
    checkpoint.config.resolved_config["dpo"]["loss_scale"] = "default"
    with pytest.raises(ValueError, match="algorithm"):
        checkpoint.load()
