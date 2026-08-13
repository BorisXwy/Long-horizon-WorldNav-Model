#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
LOG_ROOT="$NAV_ROOT/log/data_prep"
mkdir -p "$LOG_ROOT"

while tmux has-session -t nav_vbench_stats20_gpu1 2>/dev/null; do
  echo "[$(date -Is)] 等待 GPU 1 上的 VBench 流水线完成"
  sleep 60
done

echo "[$(date -Is)] 开始增量重扫和 VAE latent 缓存"
exec bash "$NAV_ROOT/scripts/run_incremental_data_prep.sh" cuda:1 0 1
