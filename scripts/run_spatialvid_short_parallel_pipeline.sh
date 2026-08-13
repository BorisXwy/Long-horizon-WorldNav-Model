#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="$NAV_ROOT/../virtual_env/.venv_infinite_world/bin/python"
ID="${1:-spatialvid-short-parallel-$(date +%Y%m%d-%H%M%S)}"
LOG_ROOT="$NAV_ROOT/log/data_prep/$ID"
STATUS="$LOG_ROOT/status.log"
NUM_SHARDS=7
GPU_MAP=(0 0 0 1 1 1 1)
mkdir -p "$LOG_ROOT"

status() {
  echo "[$(date -Is)] $*" | tee -a "$STATUS"
}

status "启动 $NUM_SHARDS 个 SpatialVID VAE shards；GPU0=3，GPU1=4"
PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU="${GPU_MAP[$SHARD]}"
  "$PYTHON" "$NAV_ROOT/scripts/cache_spatialvid_short_latents.py" \
    --device "cuda:$GPU" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD" \
    >"$LOG_ROOT/latent-shard-$(printf '%03d' "$SHARD").log" 2>&1 &
  PIDS+=("$!")
  status "shard=$SHARD gpu=$GPU pid=${PIDS[-1]}"
done

FAILED=0
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  RC=0
  wait "${PIDS[$SHARD]}" || RC=$?
  status "shard=$SHARD 完成 exit=$RC"
  if (( RC != 0 )); then FAILED=1; fi
done
if (( FAILED != 0 )); then
  status "至少一个latent shard失败，停止自动训练"
  exit 1
fi

status "全部latent shards完成，开始action/latent配对校验"
"$PYTHON" "$NAV_ROOT/scripts/verify_spatialvid_short_ready.py" \
  >"$LOG_ROOT/verify.log" 2>&1
status "配对校验通过"

while true; do
  USED="$(nvidia-smi --id=1 --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$USED" =~ ^[0-9]+$ ]] && (( USED <= 12000 )); then break; fi
  status "等待GPU 1进入训练：当前${USED:-unknown} MiB"
  sleep 60
done

status "启动A：从RE10K 1000-step checkpoint继续"
bash "$NAV_ROOT/scripts/run_spatialvid_short_training.sh" \
  latent_prefix 1 "$ID-A-latent-prefix" \
  >"$LOG_ROOT/train-A.log" 2>&1
status "A训练完成，启动B"
bash "$NAV_ROOT/scripts/run_spatialvid_short_training.sh" \
  dit_condition 1 "$ID-B-dit-condition" \
  >"$LOG_ROOT/train-B.log" 2>&1
status "B训练完成；流水线结束"
