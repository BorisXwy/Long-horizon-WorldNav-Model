#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="${PYTHON:-python}"
GPU="${1:-0}"
RUN_NAME="${2:-v1-wan-stage1-smoke-$(date +%Y%m%d-%H%M%S)}"
BATCH_SIZE="${3:-1}"
ACCUM="${4:-16}"
STEPS="${5:-3}"
TRAIN_SCOPE="${TRAIN_SCOPE:-full}"
NUM_WORKERS="${NUM_WORKERS:-2}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${NAV_ROOT}/src:${NAV_ROOT}/../Infinite-World:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON}" "${NAV_ROOT}/scripts/train_v1_wan_stage1.py" \
  --device cuda:0 \
  --run-name "${RUN_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${ACCUM}" \
  --steps "${STEPS}" \
  --train-scope "${TRAIN_SCOPE}" \
  --num-workers "${NUM_WORKERS}" \
  --log-every 1 \
  --save-every 0 \
  --smoke
