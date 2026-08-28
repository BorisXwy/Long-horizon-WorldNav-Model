#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln
RUN_ID=${1:-official_university_24f_sdpa_20260829}
OUT="$ROOT/NAV/result/streaming_3d_reproduction/abot_recon/$RUN_ID"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABOT_RECON_NO_TQDM=1

exec "$ROOT/virtual_env/.venv_abot_recon/bin/python" \
  "$ROOT/ABot-Recon/demo.py" \
  --image-dir "$ROOT/lingbot-map/example/university" \
  --checkpoint /sharedata/ABot-Recon/checkpoints/abot_recon.safetensors \
  --output-dir "$OUT" \
  --attention-backend sdpa \
  --end 24 \
  --no-loop-closure \
  --save-local-points \
  --save-confidence \
  --no-save-world-points
