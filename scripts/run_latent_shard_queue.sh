#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="$NAV_ROOT/../virtual_env/.venv_infinite_world/bin/python"
MANIFEST="/sharedata/NAV/derived/manifests/full_episodes_v1.jsonl"
LATENTS="/sharedata/NAV/derived/latents/full_episodes_v1"

GPU="${1:?用法: run_latent_shard_queue.sh <gpu> <comma-separated-shards> <run-name>}"
SHARDS_CSV="${2:?缺少 shard 列表}"
RUN_NAME="${3:?缺少 run name}"
LOG_ROOT="$NAV_ROOT/log/data_prep/$RUN_NAME"
mkdir -p "$LOG_ROOT"

status() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_ROOT/status.log"
}

IFS=',' read -r -a SHARDS <<< "$SHARDS_CSV"
for SHARD in "${SHARDS[@]}"; do
  status "GPU${GPU}启动幂等latent shard=${SHARD}"
  RC=0
  "$PYTHON" "$NAV_ROOT/scripts/cache_multidataset_latents.py" \
    --manifest "$MANIFEST" \
    --output "$LATENTS" \
    --device "cuda:${GPU}" \
    --num-shards 10 \
    --shard-index "$SHARD" \
    --min-chunks 2 \
    --max-cache-chunks 0 \
    >>"$LOG_ROOT/latent-shard-$(printf '%03d' "$SHARD").log" 2>&1 || RC=$?
  status "GPU${GPU} latent shard=${SHARD}结束 exit=${RC}"
  if ((RC != 0)); then
    exit "$RC"
  fi
done

status "GPU${GPU}单worker latent队列完成"
