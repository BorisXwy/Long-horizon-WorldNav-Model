#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
DATA_PYTHON="$NAV_ROOT/../virtual_env/.venv_data_prep/bin/python"
VAE_PYTHON="$NAV_ROOT/../virtual_env/.venv_infinite_world/bin/python"
ID="${1:-remaining-pose-$(date +%Y%m%d-%H%M%S)}"
GPU="${2:-0}"
LOG_ROOT="$NAV_ROOT/log/data_prep/$ID"
MANIFEST="/sharedata/NAV/derived/manifests/remaining_pose_short.jsonl"
LATENTS="/sharedata/NAV/derived/latents_pose_aligned"
mkdir -p "$LOG_ROOT"

status() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_ROOT/status.log"
}

status "生成 RE10K、DL3DV、Argoverse 2 pose-aligned action manifest"
"$DATA_PYTHON" "$NAV_ROOT/scripts/prepare_multidataset_manifest.py" \
  --datasets re10k,dl3dv,argoverse2 \
  --output "$MANIFEST" \
  >"$LOG_ROOT/actions.log" 2>&1
status "action manifest 完成；在 GPU $GPU 启动 short latent"

"$VAE_PYTHON" "$NAV_ROOT/scripts/cache_multidataset_latents.py" \
  --manifest "$MANIFEST" \
  --output "$LATENTS" \
  --device "cuda:$GPU" \
  --min-chunks 2 \
  --max-cache-chunks 3 \
  >"$LOG_ROOT/latents.log" 2>&1

status "latent 编码完成，开始完整性校验"
"$VAE_PYTHON" "$NAV_ROOT/scripts/verify_latent_cache.py" \
  --manifest "$MANIFEST" \
  --latents "$LATENTS" \
  --required-max-chunks 3 \
  >"$LOG_ROOT/verify.log" 2>&1
status "剩余 Pose 数据 short action/latent 全部完成"
