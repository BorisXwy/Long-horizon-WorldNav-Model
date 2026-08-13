#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
LOG_DIR="${NAV_ROOT}/log/v1_data_prep"
PID_FILE="${LOG_DIR}/build_v1_stage3_vln_raw.pid"
LOG_FILE="${LOG_DIR}/build_v1_stage3_vln_raw.log"
OUT_ROOT="${OUT_ROOT:-/sharedata/NAV/derived/v1/vln/raw_policy}"

STATUS="not_started"
if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  STATUS="running"
elif [[ -f "${LOG_FILE}" ]] && grep -q 'EXIT_CODE:0' "${LOG_FILE}"; then
  STATUS="finished"
elif [[ -f "${LOG_FILE}" ]] && grep -q 'EXIT_CODE:' "${LOG_FILE}"; then
  STATUS="failed"
fi

echo "status=${STATUS} pid_file=${PID_FILE} log=${LOG_FILE}"
if [[ -f "${LOG_FILE}" ]]; then
  tail -n 20 "${LOG_FILE}" || true
fi

if [[ -L "${OUT_ROOT}/latest" ]]; then
  RUN_ROOT="$(readlink -f "${OUT_ROOT}/latest")"
else
  RUN_ROOT="$(find "${OUT_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort | tail -1 || true)"
fi

if [[ -n "${RUN_ROOT}" && -d "${RUN_ROOT}" ]]; then
  echo "run_root=${RUN_ROOT}"
  du -sh "${RUN_ROOT}" || true
  if [[ -f "${RUN_ROOT}/summary.json" ]]; then
    python - <<PY
import json
from pathlib import Path
p=Path("${RUN_ROOT}")/"summary.json"
obj=json.load(open(p))
print("totals="+json.dumps(obj.get("totals", {}), ensure_ascii=False))
print("sources="+json.dumps(obj.get("sources", {}), ensure_ascii=False)[:2000])
PY
  else
    echo "summary=pending"
  fi
  find "${RUN_ROOT}/manifests" -maxdepth 1 -type f 2>/dev/null | sort || true
fi
