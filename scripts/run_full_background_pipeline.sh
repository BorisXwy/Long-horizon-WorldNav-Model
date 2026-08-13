#!/usr/bin/env bash
# 完整后台依赖链：VBench -> latent -> 完整性校验 -> A/B 三阶段训练。
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python"
LOG_ROOT="$NAV_ROOT/log/full_pipeline"
PIPELINE_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
STATUS_FILE="$LOG_ROOT/${PIPELINE_ID}.status"
mkdir -p "$LOG_ROOT"

status() {
  local message="$1"
  echo "[$(date -Is)] $message" | tee -a "$STATUS_FILE"
}

wait_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    status "等待 tmux 会话结束：$session"
    sleep 60
  done
}

wait_gpu_idle() {
  local gpu="$1"
  local limit_mib="${2:-8000}"
  local used
  while true; do
    used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= limit_mib )); then
      status "GPU $gpu 可用：已用 ${used} MiB"
      return
    fi
    status "等待 GPU $gpu 可用：已用 ${used:-unknown} MiB，门限 ${limit_mib} MiB"
    sleep 60
  done
}

status "流水线启动：$PIPELINE_ID"
wait_session nav_vbench_stats20_gpu1

while true; do
  status "开始重建清单并缓存缺失 latent"
  if bash "$NAV_ROOT/scripts/run_incremental_data_prep.sh" cuda:1 0 1 \
      >>"$LOG_ROOT/${PIPELINE_ID}-latent.log" 2>&1 \
    && "$PYTHON" "$NAV_ROOT/scripts/verify_latent_cache.py" \
      >>"$LOG_ROOT/${PIPELINE_ID}-latent-verify.log" 2>&1; then
    status "latent 完整性校验通过"
    break
  fi
  status "latent 尚未完整或源文件仍在下载，5 分钟后自动重扫重试"
  sleep 300
done

run_variant() {
  local variant="$1"
  local gpu="$2"
  local dependency_session="${3:-}"
  if [[ -n "$dependency_session" ]]; then
    wait_session "$dependency_session"
  fi
  wait_gpu_idle "$gpu" 8000
  status "启动 $variant 三阶段训练，GPU $gpu"
  bash "$NAV_ROOT/scripts/run_curriculum_train.sh" \
    "$variant" "$gpu" "$PIPELINE_ID" \
    >>"$LOG_ROOT/${PIPELINE_ID}-${variant}.log" 2>&1
  status "$variant 三阶段训练完成"
}

# A 等待 GPU 0 上的 Kinetics 标注结束；B 在预处理释放 GPU 1 后启动。
run_variant latent_prefix 0 kinetics_vggt_all &
PID_A=$!
run_variant dit_condition 1 &
PID_B=$!

FAILED=0
wait "$PID_A" || FAILED=1
wait "$PID_B" || FAILED=1
if (( FAILED )); then
  status "流水线失败：至少一个 variant 非正常退出"
  exit 1
fi
status "流水线全部完成"
