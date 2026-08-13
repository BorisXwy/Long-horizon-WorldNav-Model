#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
if [[ -n "${PID_FILE:-}" ]]; then
  PID_FILE="${PID_FILE}"
elif [[ -s "${NAV_ROOT}/log/v1_train/stage1_scaffold_formal_bs192.pid" ]]; then
  PID_FILE="${NAV_ROOT}/log/v1_train/stage1_scaffold_formal_bs192.pid"
else
  PID_FILE="${NAV_ROOT}/log/v1_train/stage1_scaffold.pid"
fi
RUN_DIR="${RUN_DIR:-$(ls -td "${NAV_ROOT}"/log/v1-stage1-scaffold-* 2>/dev/null | head -1 || true)}"

STATUS="not_started"
if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  STATUS="running"
elif [[ -n "${RUN_DIR}" && -f "${RUN_DIR}/stdout.log" ]] && grep -q 'EXIT_CODE:0' "${RUN_DIR}/stdout.log"; then
  STATUS="finished"
elif [[ -n "${RUN_DIR}" && -f "${RUN_DIR}/stdout.log" ]] && grep -q 'EXIT_CODE:' "${RUN_DIR}/stdout.log"; then
  STATUS="failed"
fi

echo "status=${STATUS} pid_file=${PID_FILE} run_dir=${RUN_DIR}"
if [[ -n "${RUN_DIR}" && -d "${RUN_DIR}" ]]; then
  if [[ -f "${RUN_DIR}/config.json" ]]; then
    python - <<PY
import json
p="${RUN_DIR}/config.json"
obj=json.load(open(p))
keys=["run_name","dataset_len","batch_size","steps","lr","lambda_visual","lambda_action","hidden_dim","num_layers","num_heads","register_tokens","action_horizon","trainable_parameters","dtype"]
print("config="+json.dumps({k: obj.get(k) for k in keys}, ensure_ascii=False))
PY
  fi
  if [[ -f "${RUN_DIR}/train.log" ]]; then
    echo "last_train_events:"
    grep '"event": "step"' "${RUN_DIR}/train.log" | tail -8 || true
    grep '"event": "complete"' "${RUN_DIR}/train.log" | tail -1 || true
  fi
  echo "checkpoints:"
  find "${RUN_DIR}" -maxdepth 1 -name '*.pt' -printf '%f %s\n' | sort || true
fi
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
