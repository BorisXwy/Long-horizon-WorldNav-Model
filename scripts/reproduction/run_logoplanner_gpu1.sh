#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln
RUN_ID=${1:-official_nyuv2_rgbd_12f_20260829}
OUT="$ROOT/NAV/result/streaming_3d_reproduction/logoplanner/$RUN_ID"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$ROOT/virtual_env/.venv_navdp/bin/python" \
  "$ROOT/NAV/scripts/reproduction/run_logoplanner_official.py" \
  --repo "$ROOT/NavDP" \
  --checkpoint /sharedata/LoGoPlanner/modelscope/logoplanner_policy.ckpt \
  --rgb-dir /sharedata/datasets/NYUDv2/images \
  --depth-dir /sharedata/datasets/NYUDv2/depth \
  --output-dir "$OUT" \
  --num-frames 12
