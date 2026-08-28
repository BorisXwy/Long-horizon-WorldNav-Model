#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln
RUN_ID=${1:-official_university_24f_sdpa_20260829}
OUT="$ROOT/NAV/result/streaming_3d_reproduction/lingbot_map/$RUN_ID"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$ROOT/virtual_env/.venv_lingbot_map/bin/python" \
  "$ROOT/NAV/scripts/reproduction/run_lingbot_map_official.py" \
  --repo "$ROOT/lingbot-map" \
  --checkpoint /sharedata/lingbot-map/lingbot-map.pt \
  --image-dir "$ROOT/lingbot-map/example/university" \
  --output-dir "$OUT" \
  --first-k 24 \
  --scale-frames 8 \
  --keyframe-interval 1
