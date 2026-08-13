#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
LOG_DIR="${NAV_ROOT}/log/v1_data_prep"
OUTPUT_ROOT="${OUTPUT_ROOT:-/sharedata/NAV/derived/v1/vae_packs_hdf5_fullgpu14_7x7}/workers"

for PID_FILE in "${LOG_DIR}"/encode_v1_vae_packs_worker*.pid; do
  [[ -e "${PID_FILE}" ]] || { echo "no worker pid files in ${LOG_DIR}"; exit 0; }
  BASENAME="$(basename "${PID_FILE}")"
  WORKER="${BASENAME#encode_v1_vae_packs_worker}"
  WORKER="${WORKER%.pid}"
  PID_FILE="${LOG_DIR}/encode_v1_vae_packs_worker${WORKER}.pid"
  LOG_FILE="${LOG_DIR}/encode_v1_vae_packs_worker${WORKER}.log"
  WORKER_ROOT="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "worker-$(printf '%03d' "${WORKER}")-of-*" | head -1 || true)"
  if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    STATUS="running"
  else
    STATUS="not_running"
  fi
  if [[ -z "${WORKER_ROOT}" && "${STATUS}" == "not_running" ]]; then
    continue
  fi
  COMPLETED=0
  ERRORS=0
  if [[ -f "${LOG_FILE}" ]]; then
    COMPLETED="$(grep -c '\"cached\"' "${LOG_FILE}" || true)"
    ERRORS="$(grep -c '\"error\"' "${LOG_FILE}" || true)"
  fi
  SHARDS=0
  INDEX_ROWS="0"
  if [[ -n "${WORKER_ROOT}" && -d "${WORKER_ROOT}/shards" ]]; then
    SHARDS="$(find "${WORKER_ROOT}/shards" -name 'data-*.h5' | wc -l)"
    if [[ -f "${WORKER_ROOT}/meta/info.json" ]]; then
      INDEX_ROWS="$(python - <<PY
import json
print(json.load(open("${WORKER_ROOT}/meta/info.json")).get("samples", 0))
PY
)"
    fi
  fi
  echo "worker=${WORKER} status=${STATUS} completed=${COMPLETED} errors=${ERRORS} shards=${SHARDS} indexed=${INDEX_ROWS} log=${LOG_FILE}"
done
