#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
DATA_PYTHON="$NAV_ROOT/../virtual_env/.venv_data_prep/bin/python"
TRAIN_PYTHON="$NAV_ROOT/../virtual_env/.venv_infinite_world/bin/python"
ID="${1:-spatialvid-short-$(date +%Y%m%d-%H%M%S)}"
GPU="${2:-1}"
LOG_ROOT="$NAV_ROOT/log/data_prep/$ID"
STATUS="$LOG_ROOT/status.log"
mkdir -p "$LOG_ROOT"

status() {
  echo "[$(date -Is)] $*" | tee -a "$STATUS"
}

status "并行启动 SpatialVID action 与 short latent"
"$DATA_PYTHON" "$NAV_ROOT/scripts/prepare_spatialvid_short_actions.py" \
  >"$LOG_ROOT/actions.log" 2>&1 &
ACTION_PID=$!

"$TRAIN_PYTHON" "$NAV_ROOT/scripts/cache_spatialvid_short_latents.py" \
  --device "cuda:$GPU" \
  >"$LOG_ROOT/latents.log" 2>&1 &
LATENT_PID=$!

ACTION_RC=0
LATENT_RC=0
wait "$ACTION_PID" || ACTION_RC=$?
status "action 准备结束：exit=$ACTION_RC"
wait "$LATENT_PID" || LATENT_RC=$?
status "latent 准备结束：exit=$LATENT_RC"

if (( ACTION_RC != 0 || LATENT_RC != 0 )); then
  status "数据准备进程异常，停止自动训练"
  exit 1
fi

"$TRAIN_PYTHON" "$NAV_ROOT/scripts/verify_spatialvid_short_ready.py" \
  >"$LOG_ROOT/verify.log" 2>&1
status "数据配对校验通过"

while true; do
  USED="$(nvidia-smi --id="$GPU" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$USED" =~ ^[0-9]+$ ]] && (( USED <= 12000 )); then
    break
  fi
  status "等待 GPU $GPU 可用：当前 ${USED:-unknown} MiB"
  sleep 60
done

status "启动 A：从 RE10K 1000-step checkpoint 继续"
bash "$NAV_ROOT/scripts/run_spatialvid_short_training.sh" \
  latent_prefix "$GPU" "$ID-A-latent-prefix" \
  >"$LOG_ROOT/train-A.log" 2>&1
status "A short-history 训练完成"

status "启动 B：从 RE10K 1000-step checkpoint 继续"
bash "$NAV_ROOT/scripts/run_spatialvid_short_training.sh" \
  dit_condition "$GPU" "$ID-B-dit-condition" \
  >"$LOG_ROOT/train-B.log" 2>&1
status "B short-history 训练完成；流水线结束"
