#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="$NAV_ROOT/../virtual_env/.venv_infinite_world/bin/python"
RUN_NAME="stage-one-v1-dl3dv-b-10000"
TRAIN_LOG="$NAV_ROOT/log/$RUN_NAME/train.log"
LAUNCH_LOG="$NAV_ROOT/log/launchers/$RUN_NAME.log"
PREP_LOG_ROOT="$NAV_ROOT/log/data_prep/full-episodes-v1-stageone-b-coexist"
MANIFEST="/sharedata/NAV/derived/manifests/full_episodes_v1.jsonl"
LATENTS="/sharedata/NAV/derived/latents/full_episodes_v1"
MIN_FREE_MIB=45000

mkdir -p "$NAV_ROOT/log/launchers" "$PREP_LOG_ROOT"

status() {
  echo "[$(date -Is)] $*" | tee -a "$PREP_LOG_ROOT/status.log"
}

while true; do
  FREE_MIB="$(nvidia-smi --id=1 --query-gpu=memory.free \
    --format=csv,noheader,nounits | tr -d ' ')"
  if ((FREE_MIB >= MIN_FREE_MIB)); then
    break
  fi
  status "等待GPU1释放：free=${FREE_MIB}MiB，阈值=${MIN_FREE_MIB}MiB"
  sleep 60
done

status "GPU1已释放，启动Stage One 1.0 B"
bash "$NAV_ROOT/scripts/run_stage_one_v1_dl3dv.sh" \
  dit_condition 1 1 10000 "$RUN_NAME" 1000 >"$LAUNCH_LOG" 2>&1 &
TRAIN_PID="$!"
status "B pid=$TRAIN_PID"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if [[ -f "$TRAIN_LOG" ]] && grep -q '"step": 1' "$TRAIN_LOG"; then
    break
  fi
  sleep 10
done

if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  wait "$TRAIN_PID"
  exit $?
fi

status "B首步完成；在GPU1余量中启动3路latent shards 5/6/7"
PREP_PIDS=()
for SHARD in 5 6 7; do
  "$PYTHON" "$NAV_ROOT/scripts/cache_multidataset_latents.py" \
    --manifest "$MANIFEST" \
    --output "$LATENTS" \
    --device cuda:1 \
    --num-shards 10 \
    --shard-index "$SHARD" \
    --min-chunks 2 \
    --max-cache-chunks 0 \
    >"$PREP_LOG_ROOT/latent-shard-$(printf '%03d' "$SHARD").log" 2>&1 &
  PREP_PIDS+=("$!")
  status "latent shard=$SHARD gpu=1 pid=${PREP_PIDS[-1]}"
done

TRAIN_RC=0
wait "$TRAIN_PID" || TRAIN_RC=$?
status "B训练结束 exit=$TRAIN_RC"

PREP_FAILED=0
for INDEX in "${!PREP_PIDS[@]}"; do
  RC=0
  wait "${PREP_PIDS[$INDEX]}" || RC=$?
  status "latent shard=$((INDEX + 5))结束 exit=$RC"
  if ((RC != 0)); then PREP_FAILED=1; fi
done

if ((TRAIN_RC != 0 || PREP_FAILED != 0)); then
  exit 1
fi
