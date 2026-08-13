#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="$NAV_ROOT/../virtual_env/.venv_infinite_world/bin/python"
MANIFEST="/sharedata/NAV/derived/manifests/remaining_pose_short.jsonl"
LATENTS="/sharedata/NAV/derived/latents_pose_aligned/dl3dv"
LATENT_ROOT="/sharedata/NAV/derived/latents_pose_aligned"
GPU="${1:-1}"
ID="${2:-dl3dv-history-extension-test-$(date +%Y%m%d-%H%M%S)}"
STEPS="${3:-1000}"
LOG_ROOT="$NAV_ROOT/log/data_prep/$ID"
mkdir -p "$LOG_ROOT"

status() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_ROOT/status.log"
}

run_train() {
  local variant="$1"
  local resume="$2"
  local run_name="$ID-$variant"

  status "启动 $variant 全参数测试训练：2–6 chunks，${STEPS} steps"
  "$PYTHON" "$NAV_ROOT/scripts/train_infinite_register.py" \
    --variant "$variant" \
    --latent-cache "$LATENTS" \
    --cache-manifest "$MANIFEST" \
    --resume-checkpoint "$resume" \
    --train-scope full \
    --device "cuda:$GPU" \
    --run-name "$run_name" \
    --steps "$STEPS" \
    --save-every 250 \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --min-chunks 2 \
    --max-chunks 6 \
    --shuffle-seed 20260729 \
    --plateau-window 0 \
    >"$LOG_ROOT/train-$variant.log" 2>&1
  status "$variant 测试训练完成"
}

status "增量补齐 DL3DV latent 至最多6 chunks；源数据当前最长5 chunks"
"$PYTHON" "$NAV_ROOT/scripts/cache_multidataset_latents.py" \
  --manifest "$MANIFEST" \
  --output "$LATENT_ROOT" \
  --datasets dl3dv \
  --device "cuda:$GPU" \
  --min-chunks 2 \
  --max-cache-chunks 6 \
  >"$LOG_ROOT/extend-latents.log" 2>&1
status "DL3DV latent 补齐完成"

run_train latent_prefix \
  "$NAV_ROOT/log/re10k-all-a-full-ebs4-from-infinite/full-final.pt"
run_train dit_condition \
  "$NAV_ROOT/log/re10k-all-b-full-ebs4-from-infinite/full-final.pt"

status "A/B DL3DV history-extension 测试全部完成"
