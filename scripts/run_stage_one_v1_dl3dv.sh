#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="$NAV_ROOT/../virtual_env/.venv_infinite_world/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LATENTS="/sharedata/NAV/derived/latents/full_episodes_v1/dl3dv"
ACTIONS="/sharedata/NAV/derived/actions/full_episodes_v1/dl3dv"
MANIFEST="/sharedata/NAV/derived/manifests/full_episodes_v1.jsonl"

VARIANT="${1:?用法: run_stage_one_v1_dl3dv.sh <latent_prefix|dit_condition> [gpu] [physical_batch] [steps] [run_name] [save_every] [gradient_accumulation_steps]}"
GPU="${2:-1}"
BATCH="${3:-1}"
STEPS="${4:-10000}"
RUN_NAME="${5:-stage-one-v1-dl3dv-${VARIANT}-bs${BATCH}-$(date +%Y%m%d-%H%M%S)}"
SAVE_EVERY="${6:-1000}"
GRAD_ACCUM="${7:-1}"

exec "$PYTHON" "$NAV_ROOT/scripts/train_infinite_register.py" \
  --variant "$VARIANT" \
  --latent-cache "$LATENTS" \
  --action-cache "$ACTIONS" \
  --cache-manifest "$MANIFEST" \
  --train-scope full \
  --device "cuda:$GPU" \
  --run-name "$RUN_NAME" \
  --steps "$STEPS" \
  --save-every "$SAVE_EVERY" \
  --batch-size "$BATCH" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --history-lengths 1,2,3 \
  --min-chunks 2 \
  --max-chunks 4 \
  --shuffle-seed 20260730 \
  --plateau-window 0
