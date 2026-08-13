#!/usr/bin/env bash
# 可重复运行：重新扫描新增 episode，只缓存尚不存在的 latent。
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
DATA_PYTHON="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_data_prep/bin/python"
INFINITE_PYTHON="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python"
DEVICE="${1:-cuda:1}"
SHARD_INDEX="${2:-0}"
NUM_SHARDS="${3:-1}"

"$DATA_PYTHON" "$NAV_ROOT/scripts/prepare_multidataset_manifest.py"
"$INFINITE_PYTHON" "$NAV_ROOT/scripts/cache_multidataset_latents.py" \
  --device "$DEVICE" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --min-chunks 2
