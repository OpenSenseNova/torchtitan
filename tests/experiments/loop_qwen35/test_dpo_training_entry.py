# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from unittest.mock import patch

from torchtitan.experiments.loop_qwen35.train import parse_args


def args(*extra):
    return [
        "--model",
        "model",
        "--cached-dataset",
        "pairs",
        "--output-dir",
        "output",
        "--tp",
        "1",
        "--max-length",
        "128",
        "--loss-chunks",
        "4",
        *extra,
    ]


def test_mode_specific_defaults():
    required = [
        "--model",
        "model",
        "--cached-dataset",
        "pairs",
        "--output-dir",
        "output",
    ]
    with patch.dict("os.environ", WORLD_SIZE="1"):
        dpo = parse_args([*required, "--rlhf-type", "dpo"])
        assert dpo.tp == dpo.pp == 1
        assert not dpo.packing and dpo.packing_strategy == "greedy"
        assert dpo.dpo_token_budget == 2 * dpo.max_length and dpo.beta == 0.1
        assert dpo.decay_ratio == 0.0 and dpo.loss_scale == "default"
        sft = parse_args(required)
        assert sft.tp == sft.pp == 1 and sft.decay_ratio == 1.0
        assert sft.packing and sft.packing_strategy == "greedy"


def test_explicit_dpo_options_are_preserved():
    with patch.dict("os.environ", WORLD_SIZE="1"):
        dpo = parse_args(
            args(
                "--rlhf-type",
                "dpo",
                "--beta",
                "0.2",
                "--decay-ratio",
                "0.4",
                "--packing",
                "true",
            )
        )
    assert dpo.beta == 0.2 and dpo.decay_ratio == 0.4
    assert dpo.packing and dpo.packing_strategy == "greedy"
    assert dpo.dpo_token_budget == 256


@pytest.mark.parametrize(
    "extra",
    [
        ["--loss-scale", "all"],
        ["--beta", "nan"],
        ["--beta", "0"],
        ["--beta", "-1"],
        ["--loss-type", "ipo"],
        ["--label-smoothing", "0.1"],
        ["--packing-strategy", "balanced"],
        ["--packing-length-cache", "sft.npz"],
        ["--packing-prefetch-packs", "auto"],
        ["--dpo-token-budget", "7"],
        ["--dpo-token-budget", "-8"],
        ["--dpo-token-budget", "0"],
        ["--cp", "0"],
        ["--initial-checkpoint", "hf", "--initial-rank-weights", "rank"],
    ],
)
def test_reject_unsupported_dpo(extra):
    with patch.dict("os.environ", WORLD_SIZE="1"), pytest.raises(SystemExit):
        parse_args(args("--rlhf-type", "dpo", *extra))


def test_loop_count_overrides_eval_mode():
    import torch
    from test_loop_semantics import AffineLayer, affine_model

    model = affine_model([AffineLayer(i) for i in range(8)])
    model.loop_settings.eval_count = 3
    inputs = torch.ones(4, 1)
    model.train()
    expected = model(inputs, loop_count=1)
    model.eval()
    actual = model(inputs, loop_count=1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, model(inputs))
    with pytest.raises(ValueError, match="loop_count"):
        model(inputs, loop_count=0)


def test_prepare_step_keeps_token_units():
    from unittest.mock import Mock

    from torchtitan.experiments.loop_qwen35.dpo.engine import DPOTrainingEngine

    engine = DPOTrainingEngine.__new__(DPOTrainingEngine)
    engine.prepare_step = Mock()
    engine.prepare_dpo_step(3, 127, num_accumulation_steps=2)
    engine.prepare_step.assert_called_once_with(127, num_accumulation_steps=2)
    with pytest.raises(ValueError, match="valid pairs"):
        engine.prepare_dpo_step(0, 0, num_accumulation_steps=1)


def test_explicit_reference_rejects_different_tokenizer(tmp_path):
    from types import SimpleNamespace

    from torchtitan.experiments.loop_qwen35.dpo.config import configure_dpo

    policy, reference = tmp_path / "policy", tmp_path / "reference"
    for folder, content in [(policy, "policy"), (reference, "different")]:
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "tokenizer.json").write_text(content)
    options = SimpleNamespace(
        ref_model=str(reference), ref_checkpoint=None, model=str(policy), max_length=128
    )
    base = dict(name="fixture")
    with patch(
        "torchtitan.experiments.loop_qwen35.model_config.identify_base",
        return_value=("fixture", base),
    ):
        with pytest.raises(ValueError, match="tokenizer/processor/template mismatch"):
            configure_dpo(SimpleNamespace(), options, dict(base_model=base))


@pytest.mark.parametrize("option", ["--loss-scale", "--loss_scale"])
def test_explicit_last_round_and_sft_isolation(option):
    with patch.dict("os.environ", WORLD_SIZE="1"):
        assert (
            parse_args(args("--rlhf-type", "dpo", option, "last_round")).loss_scale
            == "last_round"
        )
        with pytest.raises(SystemExit):
            parse_args(args(option, "default"))
