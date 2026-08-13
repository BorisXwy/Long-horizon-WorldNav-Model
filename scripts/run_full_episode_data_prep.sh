#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
DATA_PYTHON="$NAV_ROOT/../virtual_env/.venv_data_prep/bin/python"
VAE_PYTHON="$NAV_ROOT/../virtual_env/.venv_infinite_world/bin/python"
ID="${1:-full-episodes-v1-$(date +%Y%m%d-%H%M%S)}"
LOG_ROOT="$NAV_ROOT/log/data_prep/$ID"
MANIFEST="/sharedata/NAV/derived/manifests/full_episodes_v1.jsonl"
ACTIONS="/sharedata/NAV/derived/actions/full_episodes_v1"
LATENTS="/sharedata/NAV/derived/latents/full_episodes_v1"
# GPU 1 当前由导航评测服务占用。先用 GPU 0 三路稳定编码；所有输出按
# sample_id 幂等，GPU 1 释放后可安全增加 shard 数断点续跑。
NUM_SHARDS=10
GPU_MAP=(0 0 0 0 0 1 1 1 1 1)
mkdir -p "$LOG_ROOT"

status() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_ROOT/status.log"
}

if [[ ! -s "$MANIFEST" ]]; then
  status "构建完整 episode manifest"
  "$DATA_PYTHON" "$NAV_ROOT/scripts/prepare_multidataset_manifest.py" \
    --output "$MANIFEST" \
    --datasets dl3dv,re10k,spatialvid,argoverse2 \
    >"$LOG_ROOT/manifest.log" 2>&1
else
  status "复用已有完整 episode manifest"
fi

if [[ ! -f "$ACTIONS/_COMPLETE" ]]; then
  status "导出完整时间轴 dense Action"
  "$DATA_PYTHON" "$NAV_ROOT/scripts/export_full_episode_actions.py" \
    --manifest "$MANIFEST" \
    --output "$ACTIONS" \
    >"$LOG_ROOT/actions.log" 2>&1
else
  status "复用已有完整时间轴 dense Action"
fi

status "启动完整 episode latent：10 shards；GPU0=5，GPU1=5"
DL3DV_READY="/sharedata/NAV/derived/stage_ready/full_episodes_v1_dl3dv.ready"
rm -f "$DL3DV_READY"
(
  EXPECTED="$("$DATA_PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
print(sum(
    json.loads(line)["dataset"] == "dl3dv"
    for line in open(sys.argv[1], encoding="utf-8")
))
PY
)"
  while true; do
    READY="$(find "$LATENTS/dl3dv" -maxdepth 1 -type f \
      -name '*.pt' 2>/dev/null | wc -l)"
    if ((READY >= EXPECTED)); then
      touch "$DL3DV_READY"
      status "DL3DV完整latent已就绪：${READY}/${EXPECTED}；可启动长程验证"
      break
    fi
    sleep 60
  done
) &
READY_MONITOR_PID="$!"

PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU="${GPU_MAP[$SHARD]}"
  "$VAE_PYTHON" "$NAV_ROOT/scripts/cache_multidataset_latents.py" \
    --manifest "$MANIFEST" \
    --output "$LATENTS" \
    --device "cuda:$GPU" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD" \
    --min-chunks 2 \
    --max-cache-chunks 0 \
    >"$LOG_ROOT/latent-shard-$(printf '%03d' "$SHARD").log" 2>&1 &
  PIDS+=("$!")
  status "latent shard=$SHARD gpu=$GPU pid=${PIDS[-1]}"
done

FAILED=0
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  RC=0
  wait "${PIDS[$SHARD]}" || RC=$?
  status "latent shard=$SHARD 完成 exit=$RC"
  if ((RC != 0)); then FAILED=1; fi
done
if ((FAILED)); then
  kill "$READY_MONITOR_PID" 2>/dev/null || true
  status "至少一个 shard 失败；保留全部已完成 cache 供断点重启"
  exit 1
fi
wait "$READY_MONITOR_PID"

status "校验所有完整 episode latent"
"$VAE_PYTHON" "$NAV_ROOT/scripts/verify_latent_cache.py" \
  --manifest "$MANIFEST" \
  --latents "$LATENTS" \
  --required-max-chunks 0 \
  >"$LOG_ROOT/verify.log" 2>&1
status "完整 episode Action/latent 全部完成"
