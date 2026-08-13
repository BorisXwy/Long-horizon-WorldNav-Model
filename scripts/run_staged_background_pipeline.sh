#!/usr/bin/env bash
# 分阶段流水线：短缓存完成即训练 A，同时增量扩展中/长缓存。
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
DATA_PYTHON="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_data_prep/bin/python"
PYTHON="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python"
ID="${1:-staged-$(date +%Y%m%d-%H%M%S)}"
LOG_ROOT="$NAV_ROOT/log/full_pipeline"
READY_ROOT="/sharedata/NAV/derived/stage_ready/$ID"
STATUS="$LOG_ROOT/$ID.status"
mkdir -p "$LOG_ROOT" "$READY_ROOT"

status() { echo "[$(date -Is)] $*" | tee -a "$STATUS"; }
wait_file() {
  while [[ ! -f "$1" ]]; do sleep 30; done
}
wait_session() {
  while tmux has-session -t "$1" 2>/dev/null; do
    status "等待会话结束：$1"
    sleep 60
  done
}
wait_gpu() {
  local gpu="$1" used
  while true; do
    used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 8000 )); then return; fi
    status "等待 GPU $gpu：已用 ${used:-unknown} MiB"
    sleep 60
  done
}

prepare_stage() {
  local label="$1" max_chunks="$2"
  while true; do
    status "开始 $label latent：最多 $max_chunks chunks"
    "$DATA_PYTHON" "$NAV_ROOT/scripts/prepare_multidataset_manifest.py" \
      >>"$LOG_ROOT/$ID-$label-latent.log" 2>&1
    if "$PYTHON" "$NAV_ROOT/scripts/cache_multidataset_latents.py" \
        --device cuda:1 --min-chunks 2 --max-cache-chunks "$max_chunks" \
        >>"$LOG_ROOT/$ID-$label-latent.log" 2>&1 \
      && "$PYTHON" "$NAV_ROOT/scripts/verify_latent_cache.py" \
        --required-max-chunks "$max_chunks" \
        >>"$LOG_ROOT/$ID-$label-verify.log" 2>&1; then
      touch "$READY_ROOT/$label"
      status "$label latent 完成并通过校验"
      return
    fi
    status "$label latent 未完整，5 分钟后增量重扫"
    sleep 300
  done
}

train_a() {
  wait_file "$READY_ROOT/short"
  wait_session kinetics_vggt_all
  wait_gpu 0
  status "A Stage 1 启动"
  bash "$NAV_ROOT/scripts/run_train_stage.sh" latent_prefix 0 \
    "$ID-latent_prefix-short-context" 2 3 20000 2000 \
    >>"$LOG_ROOT/$ID-A-short.log" 2>&1

  wait_file "$READY_ROOT/medium"
  status "A Stage 2 启动"
  bash "$NAV_ROOT/scripts/run_train_stage.sh" latent_prefix 0 \
    "$ID-latent_prefix-medium-context" 4 7 15000 2000 \
    "$NAV_ROOT/log/$ID-latent_prefix-short-context/full-final.pt" \
    >>"$LOG_ROOT/$ID-A-medium.log" 2>&1

  wait_file "$READY_ROOT/long"
  status "A Stage 3 启动"
  bash "$NAV_ROOT/scripts/run_train_stage.sh" latent_prefix 0 \
    "$ID-latent_prefix-long-context" 8 0 10000 1500 \
    "$NAV_ROOT/log/$ID-latent_prefix-medium-context/full-final.pt" \
    >>"$LOG_ROOT/$ID-A-long.log" 2>&1
  status "A 全部完成"
}

status "分阶段流水线启动：$ID"
train_a &
PID_A=$!

prepare_stage short 3
prepare_stage medium 7
prepare_stage long 0

# GPU 1 在完整缓存完成后执行 B；A 通常已在 GPU 0 并行推进。
wait_gpu 1
status "B 三阶段训练启动"
bash "$NAV_ROOT/scripts/run_curriculum_train.sh" dit_condition 1 "$ID" \
  >>"$LOG_ROOT/$ID-B.log" 2>&1
status "B 全部完成"

wait "$PID_A"
status "分阶段流水线全部完成"
