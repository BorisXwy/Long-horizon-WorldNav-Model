#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python"
LATENTS="/sharedata/NAV/derived/latents"
VARIANT="${1:?用法: run_curriculum_train.sh latent_prefix|dit_condition GPU EXPERIMENT_PREFIX}"
GPU="${2:-0}"
PREFIX="${3:-multidata-$(date +%Y%m%d-%H%M%S)}"

case "$VARIANT" in
  latent_prefix|dit_condition) ;;
  *) echo "未知 variant: $VARIANT" >&2; exit 2 ;;
esac

run_stage() {
  local stage="$1"
  local min_chunks="$2"
  local max_chunks="$3"
  local steps="$4"
  local min_plateau="$5"
  local resume="${6:-}"
  local run_name="${PREFIX}-${VARIANT}-${stage}"
  local args=(
    "$NAV_ROOT/scripts/train_infinite_register.py"
    --variant "$VARIANT"
    --latent-cache "$LATENTS"
    --train-scope full
    --device "cuda:${GPU}"
    --run-name "$run_name"
    --steps "$steps"
    --save-every 500
    --batch-size 1
    --gradient-accumulation-steps 4
    --min-chunks "$min_chunks"
    --max-chunks "$max_chunks"
    --shuffle-seed 20260727
    --plateau-window 200
    --plateau-patience 3
    --plateau-min-relative-improvement 0.01
    --min-steps-before-plateau "$min_plateau"
  )
  if [[ -n "$resume" ]]; then
    args+=(--resume-checkpoint "$resume")
  fi
  "$PYTHON" "${args[@]}"
}

# Stage 1 不传 resume：A/B 都独立加载原始 InfiniteWorld，再构造各自 Register。
run_stage short-context 2 3 20000 2000
STAGE1="$NAV_ROOT/log/${PREFIX}-${VARIANT}-short-context/full-final.pt"

run_stage medium-context 4 7 15000 2000 "$STAGE1"
STAGE2="$NAV_ROOT/log/${PREFIX}-${VARIANT}-medium-context/full-final.pt"

run_stage long-context 8 0 10000 1500 "$STAGE2"
