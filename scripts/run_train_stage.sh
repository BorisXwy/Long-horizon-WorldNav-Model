#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python"
VARIANT="${1:?variant}"
GPU="${2:?gpu}"
RUN_NAME="${3:?run name}"
MIN_CHUNKS="${4:?min chunks}"
MAX_CHUNKS="${5:?max chunks}"
STEPS="${6:?steps}"
MIN_PLATEAU="${7:?min plateau}"
RESUME="${8:-}"

ARGS=(
  "$NAV_ROOT/scripts/train_infinite_register.py"
  --variant "$VARIANT"
  --latent-cache /sharedata/NAV/derived/latents
  --train-scope full
  --device "cuda:$GPU"
  --run-name "$RUN_NAME"
  --steps "$STEPS"
  --save-every 500
  --batch-size 1
  --gradient-accumulation-steps 4
  --min-chunks "$MIN_CHUNKS"
  --max-chunks "$MAX_CHUNKS"
  --shuffle-seed 20260727
  --plateau-window 200
  --plateau-patience 3
  --plateau-min-relative-improvement 0.01
  --min-steps-before-plateau "$MIN_PLATEAU"
)
if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume-checkpoint "$RESUME")
fi
exec "$PYTHON" "${ARGS[@]}"
