#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="${PYTHON:-python}"
GPU="${1:-0}"
RUN_NAME="${2:-v1-stage1-scaffold-smoke-$(date +%Y%m%d-%H%M%S)}"
BATCH_SIZE="${3:-1}"
STEPS="${4:-20}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
REGISTER_TOKENS="${REGISTER_TOKENS:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${NAV_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON}" "${NAV_ROOT}/scripts/train_v1_stage1_scaffold.py" \
  --device cuda:0 \
  --run-name "${RUN_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --num-heads "${NUM_HEADS}" \
  --register-tokens "${REGISTER_TOKENS}" \
  --num-workers "${NUM_WORKERS}" \
  --log-every 1 \
  --smoke
