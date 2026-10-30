# SenseNova-Loop-9B-Preview regression tests

Run from the repository root with the SenseNova-Loop-9B-Preview extra installed. CPU tests
use local synthetic tokenizers and independent tensor oracles; no download or
Swift checkout is needed.

    python -m pytest tests/experiments/SenseNova_Loop_9B_Preview -q

The suite checks encoding and binary supervision masks, packed document
boundaries, distributed normalization, DPO log-probabilities/gradients/reference
immutability, MTP weight contracts/targets/gradients/freezing, and checkpoint
state validation. Distributed numerical probes launch their own CPU workers.

## Eight-GPU SFT and MTP

Use a local original Qwen3.5-9B snapshot. The command keeps four backbone layers,
all original hidden dimensions, Loop2/mHC2 and an 8K token microbatch:

    MODEL=/models/Qwen3.5-9B
    torchrun --standalone --nproc_per_node 8 tests/experiments/SenseNova_Loop_9B_Preview/run_regression.py --num-layers 4 -- \
      --model "$MODEL" --dataset examples/train/SenseNova-Loop-9B-Preview/data/sft.jsonl \
      --loop-config examples/train/SenseNova-Loop-9B-Preview/loop2_4layers.json \
      --mhc true --mhc-start 0 --mhc-end 4 --mhc-streams 2 \
      --cp 4 --max-length 8192 --micro-batch-size 1 --global-batch-size 2 \
      --packing true --packing-strategy greedy --loss-chunks 8 \
      --activation-checkpoint true --activation-checkpoint-policy full \
      --deterministic true --gdn-output-kernel stable --flash-attention-impl FA2 \
      --train-iters 10 --save true --save-steps 5 --resume false \
      --diagnostics-interval 0 --output-dir output/sft-regression

The wrapper rejects Swift imports and records per-rank gradient samples and
norms after every update. Production metrics and complete checkpoints provide
the remaining evidence.

For MTP, use mtp_smoke.py in the same command and append --mtp-num-layers 1
--mtp-training-mode mtp_only (then repeat with joint in a fresh output directory).
The source must contain the pretrained MTP layer. Every step checks a finite,
positive MTP gradient and absent frozen gradients; the final step hashes every
parameter to verify MTP updates and frozen weights. Joint mode must also update
the backbone. The wrapper writes verification-rank-*.json.

## Eight-GPU DPO

Create a small architecture with real tokenizer/processor assets. Preparation
is a test fixture generator, never a requirement for ordinary training:

    python tests/experiments/SenseNova_Loop_9B_Preview/dpo_smoke.py prepare \
      --processor-assets "$MODEL" --output output/dpo-fixture --streams 2

    torchrun --standalone --nproc_per_node 8 tests/experiments/SenseNova_Loop_9B_Preview/dpo_smoke.py \
      --rlhf-type dpo --model output/dpo-fixture/model --dataset output/dpo-fixture/pairs \
      --mhc true --mhc-start 2 --mhc-end 4 --mhc-streams 2 \
      --cp 2 --max-length 8192 --dpo-token-budget 16384 \
      --micro-batch-size 1 --global-batch-size 4 --packing true \
      --loss-chunks 8 --activation-checkpoint true --activation-checkpoint-policy full \
      --deterministic true --gdn-output-kernel stable --flash-attention-impl FA2 \
      --train-iters 10 --save true --save-steps 5 --resume false \
      --diagnostics-interval 0 --output-dir output/dpo-regression

This tests randomized training Loop counts, fixed evaluation counts, uneven pair
tails, the real CLI and full checkpoints. It checks the reference is unchanged,
the policy updates, frozen parameters stay unchanged and Swift is never imported.
For real HF initialization, use the public DPO shell script with the four-layer
overrides in the training guide; the test-only debug architecture is not exposed
by the production CLI.

## Exact comparison and restart

Run a deterministic reference and candidate with identical arguments, seed,
hardware, kernel settings and warm caches. Use fresh output directories.
For SFT/MTP numerical evidence:

    python tests/experiments/SenseNova_Loop_9B_Preview/compare_runs.py \
      --left output/reference --right output/candidate --output output/numerics.json

Compare every model and optimizer tensor, scheduler, step counters and device
RNG on every rank:

    torchrun --standalone --nproc_per_node 8 tests/experiments/SenseNova_Loop_9B_Preview/compare_checkpoints.py \
      --left output/reference/checkpoints/step-10 \
      --right output/candidate/checkpoints/step-10 --output output/checkpoints.json

For restart, keep the same training arguments and total step count, use a new
output directory, and append --resume true --resume-checkpoint
output/candidate/checkpoints/step-5. Compare steps 6-10 using --start-step 6,
and compare complete checkpoints with --compare-data --compare-cpu-rng.
No tolerance is allowed in these exact comparisons. Data fingerprints may
intentionally change across source refactors; they must match across a restart.

The synthetic fixtures validate execution and numerical contracts, not model
quality or convergence. Full-depth, long-running and different-hardware jobs
need separate validation.
