#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
MANIFEST="/sharedata/NAV/derived/v1/manifests/stage1_video_packs.jsonl"
OUTPUT_ROOT="${OUTPUT_ROOT:-/sharedata/NAV/derived/v1/vae_packs_hdf5_fullgpu14_7x7}"
LOG_DIR="${NAV_ROOT}/log/v1_data_prep"
GPUS="${GPUS:-0,1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-6}"
WORKERS_BY_GPU="${WORKERS_BY_GPU:-0:7,1:7}"
SAMPLES_PER_HDF5_SHARD="${SAMPLES_PER_HDF5_SHARD:-256}"

mkdir -p "${OUTPUT_ROOT}/workers" "${LOG_DIR}"

GPU_LIST=()
GPU_WORKERS=()
if [[ -n "${WORKERS_BY_GPU}" ]]; then
  IFS=',' read -r -a SPECS <<< "${WORKERS_BY_GPU}"
  for SPEC in "${SPECS[@]}"; do
    GPU_LIST+=("${SPEC%%:*}")
    GPU_WORKERS+=("${SPEC##*:}")
  done
else
  IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
  for _GPU in "${GPU_LIST[@]}"; do
    GPU_WORKERS+=("${WORKERS_PER_GPU}")
  done
fi
TOTAL_WORKERS=0
for COUNT in "${GPU_WORKERS[@]}"; do
  TOTAL_WORKERS=$((TOTAL_WORKERS + COUNT))
done

echo "V1 VAE pack encode: gpus=${GPU_LIST[*]} workers=${GPU_WORKERS[*]} total_workers=${TOTAL_WORKERS} output=${OUTPUT_ROOT}"

WORKER=0
for GPU_OFFSET in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$GPU_OFFSET]}"
  COUNT="${GPU_WORKERS[$GPU_OFFSET]}"
  for LOCAL_WORKER in $(seq 0 $((COUNT - 1))); do
  WORKER_ROOT="${OUTPUT_ROOT}/workers/worker-$(printf '%03d' "${WORKER}")-of-$(printf '%03d' "${TOTAL_WORKERS}")"
  LOG_FILE="${LOG_DIR}/encode_v1_vae_packs_worker${WORKER}.log"
  PID_FILE="${LOG_DIR}/encode_v1_vae_packs_worker${WORKER}.pid"
  if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "worker ${WORKER} already running: pid $(cat "${PID_FILE}")"
    WORKER=$((WORKER + 1))
    continue
  fi
  COMMAND="CUDA_VISIBLE_DEVICES=${GPU} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ${NAV_ROOT}/scripts/datasets/encode_v1_vae_packs.py \
    --manifest "${MANIFEST}" \
    --output "${WORKER_ROOT}" \
    --format hdf5 \
    --device cuda:0 \
    --num-shards "${TOTAL_WORKERS}" \
    --shard-index "${WORKER}" \
    --samples-per-hdf5-shard "${SAMPLES_PER_HDF5_SHARD}""
  SESSION="nav_v1_vae_worker${WORKER}"
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND} > ${LOG_FILE} 2>&1; echo EXIT_CODE:\$? >> ${LOG_FILE}'"
  tmux display-message -p -t "${SESSION}" '#{pane_pid}' > "${PID_FILE}"
  echo "started worker ${WORKER}: gpu ${GPU} local_worker ${LOCAL_WORKER} tmux ${SESSION} pane_pid $(cat "${PID_FILE}") log ${LOG_FILE}"
  WORKER=$((WORKER + 1))
  done
done
