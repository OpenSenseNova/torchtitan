# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for diagnostic sampling, HSDP resume and profiling."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


import torch
from torchtitan.experiments.SenseNova_Loop_9B_Preview.profiling import sample_step

ROOT = Path(__file__).resolve().parents[3]


class OfflineTests(unittest.TestCase):
    def test_sampling_does_not_change_router_math(self):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.mhc.model import (
            MHCRouter,
        )

        router = MHCRouter(8, 4, pre_scale=0.5)
        x = torch.randn(5, 4, 8)
        first = router(x)
        self.assertTrue(hasattr(router, "last_diagnostics"))
        router.collect_diagnostics = False
        del router.last_diagnostics
        second = router(x)
        self.assertFalse(hasattr(router, "last_diagnostics"))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(sample_step(1, 0))
        self.assertFalse(sample_step(1, 2))
        self.assertTrue(sample_step(2, 2))

    def test_hsdp_cli_and_invalid_combinations(self):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.train import (
            parse_args,
        )

        base = [
            "--model",
            "unused",
            "--cached-dataset",
            "unused",
            "--output-dir",
            "unused",
            "--tp",
            "1",
            "--cp",
            "4",
            "--global-batch-size",
            "128",
            "--decay-ratio",
            "1",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "128"}):
            args = parse_args(base + ["--dp-replicate", "16", "--dp-shard", "2"])
            self.assertEqual(args.dp_replicate * args.dp_shard * args.cp, 128)
            self.assertEqual(args.packing_strategy, "greedy")
            for extra in (
                ["--dp-replicate", "3"],
                ["--diagnostics-interval", "-1"],
                ["--profile-ranks", "128"],
                ["--resume", "false", "--resume-checkpoint", "unused"],
                ["--tp", "2"],
                ["--benchmark", "true"],
                ["--activation-checkpoint-segment-size", "4"],
            ):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    parse_args(base + extra)

    def test_cpu_profiler_closes_and_exports(self):
        from torchtitan.experiments.SenseNova_Loop_9B_Preview.profiling import (
            StepProfiler,
        )

        with tempfile.TemporaryDirectory() as directory:
            profiler = StepProfiler(
                directory=directory,
                rank=0,
                ranks=[0],
                start_step=2,
                num_steps=1,
                device_type="cpu",
            )
            profiler.begin(1)
            self.assertIsNone(profiler.profiler)
            profiler.begin(2)
            torch.ones(3).square()
            profiler.end(2)
            profiler.close()
            self.assertTrue((Path(directory) / "rank-0-steps-2-2.json").is_file())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
