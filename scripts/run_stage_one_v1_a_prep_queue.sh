#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="$NAV_ROOT/../virtual_env/.venv_infinite_world/bin/python"
MANIFEST="/sharedata/NAV/derived/manifests/full_episodes_v1.jsonl"
LATENTS="/sharedata/NAV/derived/latents/full_episodes_v1"
LOG_ROOT="$NAV_ROOT/log/data_prep/full-episodes-v1-stageone-a-coexist"
EXISTING_WORKER_PID="${1:-3131260}"
mkdir -p "$LOG_ROOT"

status() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_ROOT/status.log"
}

while kill -0 "$EXISTING_WORKER_PID" 2>/dev/null; do
  status "等待现有GPU0 latent worker pid=$EXISTING_WORKER_PID 完成"
  sleep 60
done

# GPU0训练A时只允许一个VAE worker。各shard幂等续传，已经完成的样本会跳过。
for SHARD in 1 2 3 4 8 9; do
  status "GPU0启动latent shard=$SHARD"
  RC=0
  "$PYTHON" "$NAV_ROOT/scripts/cache_multidataset_latents.py" \
    --manifest "$MANIFEST" \
    --output "$LATENTS" \
    --device cuda:0 \
    --num-shards 10 \
    --shard-index "$SHARD" \
    --min-chunks 2 \
    --max-cache-chunks 0 \
    >"$LOG_ROOT/latent-shard-$(printf '%03d' "$SHARD").log" 2>&1 || RC=$?
  status "GPU0 latent shard=$SHARD结束 exit=$RC"
  if ((RC != 0)); then
    exit "$RC"
  fi
done

status "GPU0 Stage One共存latent队列完成"
