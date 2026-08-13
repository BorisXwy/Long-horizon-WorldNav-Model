#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
VGGT_PYTHON="/mnt/pool1/sharehome/xiewenyuan/.conda/envs/vggt/bin/python"
RESULT_ROOT="${NAV_ROOT}/result/kinetics_vggt_camera_all"
OUTPUT="${RESULT_ROOT}/metrics"
SHARD_INDEX="${1:-0}"
NUM_SHARDS="${2:-1}"
DEVICE="${3:-cuda:0}"

mkdir -p "${OUTPUT}" "${RESULT_ROOT}/logs"
exec "${VGGT_PYTHON}" "${NAV_ROOT}/scripts/analyze_kinetics_vggt_camera.py" \
  --all \
  --frames 8 \
  --root /sharedata/datasets/kinetics400 \
  --output "${OUTPUT}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --device "${DEVICE}"
