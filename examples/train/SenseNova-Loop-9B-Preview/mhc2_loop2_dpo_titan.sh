#!/usr/bin/env bash
# Activate an environment with TorchTitan's SenseNova-Loop-9B-Preview extra before launching.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local model directory or Hugging Face repository ID}"
DATASET="${DATASET:-${SCRIPT_DIR}/data/dpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/dpo_$(date +%Y%m%dT%H%M%S)}"
LOOP_CONFIG="${LOOP_CONFIG:-${SCRIPT_DIR}/mhc2_loop2.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
CP="${CP:-4}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((NNODES * NPROC_PER_NODE / CP * MICRO_BATCH_SIZE * 4))}"
LR="${LR:-5e-7}"

python -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK:-0}" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" \
    --master_port "${MASTER_PORT:-29745}" \
    -m torchtitan.experiments.SenseNova_Loop_9B_Preview.train \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --rlhf-type dpo \
    --loss-scale "${LOSS_SCALE:-default}" \
    --beta "${BETA:-0.1}" \
    --dpo-token-budget "$((2 * MAX_LENGTH * MICRO_BATCH_SIZE))" \
    --loop-config "${LOOP_CONFIG}" \
    --num-layers "${NUM_LAYERS:-0}" \
    --mhc true \
    --mhc-start "${MHC_START:-4}" \
    --mhc-end "${MHC_END:-28}" \
    --mhc-streams 2 \
    --cp "${CP}" \
    --cp-gdn-overlap false \
    --max-length "${MAX_LENGTH}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --packing true \
    --packing-strategy greedy \
    --freeze-vision true \
    --activation-checkpoint true \
    --activation-checkpoint-policy full \
    --loss-chunks "${LOSS_CHUNKS:-8}" \
    --gdn-output-kernel stable \
    --flash-attention-impl FA2 \
    --train-iters "${TRAIN_ITERS:-100}" \
    --lr "${LR}" \
    --router-lr "${ROUTER_LR:-${LR}}" \
    --router-warmup-steps 0 \
    --weight-decay "${WEIGHT_DECAY:-0.1}" \
    --warmup-ratio 0 \
    --decay-ratio 0 \
    --seed 42 \
    --resume "${RESUME:-false}" \
    --save true \
    --save-steps "${SAVE_STEPS:-50}" \
    --output-dir "${OUTPUT_DIR}" \
    --diagnostics-interval 0 \
    --report-to "${REPORT_TO:-none}" \
    "$@"
