# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Check public launcher arguments without starting distributed workers."""

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples/train/SenseNova-Loop-9B-Preview"
SCRIPTS = ("mhc2_loop2_titan.sh", "mhc2_loop2_dpo_titan.sh")


@pytest.fixture
def launcher_env(tmp_path):
    capture = tmp_path / "arguments.txt"
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$LAUNCHER_CAPTURE"\n')
    python.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("HOME", "PATH", "TMPDIR", "SYSTEMROOT")
    }
    env.update(
        PATH=str(tmp_path) + os.pathsep + env.get("PATH", os.defpath),
        LAUNCHER_CAPTURE=str(capture),
        OUTPUT_DIR=str(tmp_path / "output"),
    )
    return env, capture


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("model", ("test/model", "/models/weights with spaces"))
def test_launcher_uses_public_module_and_explicit_model(launcher_env, script, model):
    env, capture = launcher_env
    env["MODEL_PATH"] = model
    subprocess.run(["bash", str(EXAMPLES / script)], env=env, check=True)
    args = capture.read_text().splitlines()
    assert args[:2] == ["-m", "torch.distributed.run"]
    assert "torchtitan.experiments.SenseNova_Loop_9B_Preview.train" in args
    assert args[args.index("--model") + 1] == model
    assert args[args.index("--nproc_per_node") + 1] == "8"
    assert args[args.index("--nnodes") + 1] == "1"
    assert args[args.index("--cp") + 1] == "4"
    assert args[args.index("--max-length") + 1] == "8192"
    assert Path(args[args.index("--dataset") + 1]).is_file()
    assert Path(args[args.index("--loop-config") + 1]).is_file()
    if "dpo" in script:
        assert args[args.index("--rlhf-type") + 1] == "dpo"
        assert args[args.index("--loss-scale") + 1] == "default"
        assert args[args.index("--global-batch-size") + 1] == "8"
    else:
        assert "--rlhf-type" not in args
        assert args[args.index("--global-batch-size") + 1] == "2"


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("model", (None, ""))
def test_launcher_requires_model_before_starting_workers(launcher_env, script, model):
    env, capture = launcher_env
    if model is not None:
        env["MODEL_PATH"] = model
    result = subprocess.run(
        ["bash", str(EXAMPLES / script)], env=env, text=True, capture_output=True
    )
    assert result.returncode != 0
    assert "Set MODEL_PATH" in result.stderr
    assert not capture.exists()
