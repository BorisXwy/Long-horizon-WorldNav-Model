#!/usr/bin/env bash
# Run InternVLA-N1 DualVLN Habitat eval (NAV-EVL-002).
# Usage:
#   MAX_EPISODES=100 CUDA_VISIBLE_DEVICES=0 bash run_dualvln_eval.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
set -a
source "${NAV_ROOT}/config/r2r_ce_sota_reproduction.env"
set +a

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MAX_EPISODES="${MAX_EPISODES:--1}"
TAG="full"
if [[ "${MAX_EPISODES}" != "-1" && "${MAX_EPISODES}" -gt 0 ]]; then
  TAG="smoke${MAX_EPISODES}"
fi
OUT="${RESULT_ROOT}/dualvln/${TAG}_${RUN_ID}"
LOG_DIR="${LOG_ROOT}/internnav"
mkdir -p "${OUT}/metrics" "${OUT}/logs" "${LOG_DIR}" "${INTERNAV_ROOT}/checkpoints"
LOG="${LOG_DIR}/dualvln_eval_${TAG}_${RUN_ID}.log"

CFG_SRC="${INTERNAV_ROOT}/scripts/eval/configs/habitat_dual_system_cfg.py"
if [[ ! -f "${CFG_SRC}" ]]; then
  echo "[error] missing ${CFG_SRC}" >&2
  exit 1
fi
if [[ ! -d "${DUALVLN_CKPT_DIR}" ]]; then
  echo "[error] missing ${DUALVLN_CKPT_DIR}; run download_checkpoints.sh" >&2
  exit 1
fi

ln -sfn "${DUALVLN_CKPT_DIR}" "${INTERNAV_ROOT}/checkpoints/InternVLA-N1-DualVLN"

CFG="${OUT}/habitat_dual_system_cfg_run.py"
HAB_OUT="${OUT}/habitat_out"
mkdir -p "${HAB_OUT}"
export _NAV_CFG_SRC="${CFG_SRC}"
export _NAV_CFG_DST="${CFG}"
export _NAV_HAB_OUT="${HAB_OUT}"
export _NAV_MAX_EPS="${MAX_EPISODES}"
"${INTERNAV_VENV}/bin/python" - <<'PY'
import os, re
from pathlib import Path
src = Path(os.environ["_NAV_CFG_SRC"])
dst = Path(os.environ["_NAV_CFG_DST"])
hab_out = os.environ["_NAV_HAB_OUT"]
max_eps = int(os.environ["_NAV_MAX_EPS"])
text = src.read_text()
text = re.sub(r'"output_path"\s*:\s*"[^"]*"', f'"output_path": {hab_out!r}', text, count=1)
if "max_episodes" not in text:
    text = text.replace(
        '"max_steps_per_episode": 500,',
        f'"max_steps_per_episode": 500,\n        "max_episodes": {max_eps},',
    )
else:
    text = re.sub(r'"max_episodes"\s*:\s*-?\d+', f'"max_episodes": {max_eps}', text)
dst.write_text(text)
print(f"wrote {dst}")
PY

{
  echo "run_id=${RUN_ID} tag=${TAG} max_episodes=${MAX_EPISODES}"
  echo "cfg=${CFG}"
  echo "ckpt=${DUALVLN_CKPT_DIR}"
  echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
  "${INTERNAV_VENV}/bin/python" -c 'import torch,sys; print(sys.version); print(torch.__version__, torch.cuda.device_count())'
  cd "${INTERNAV_ROOT}"
  "${INTERNAV_VENV}/bin/python" scripts/eval/eval.py --config "${CFG}"
} 2>&1 | tee "${LOG}"

cp -a "${LOG}" "${OUT}/logs/" || true
if [[ -f "${HAB_OUT}/progress.json" ]]; then
  cp -a "${HAB_OUT}/progress.json" "${OUT}/metrics/" || true
fi
cat > "${OUT}/metrics/summary.json" <<EOF
{
  "method": "InternVLA-N1-DualVLN",
  "run_id": "${RUN_ID}",
  "tag": "${TAG}",
  "split": "${VLN_SPLIT}",
  "max_episodes": ${MAX_EPISODES},
  "checkpoint": "${DUALVLN_HF_REPO}",
  "status": "finished_or_see_log",
  "source": "Reproduced by us",
  "reported_r2r_sr_spl": [64.3, 58.5],
  "floor_r2r_sr": ${STREAMVLN_FLOOR_R2R_SR},
  "log": "${LOG}",
  "output": "${OUT}"
}
EOF

echo "Results: ${OUT}"
echo "Log: ${LOG}"
