#!/usr/bin/env bash
# Smoke / full StreamVLN VLN-CE eval (NAV-EVL-002).
# Usage:
#   MAX_EPISODES=1 CUDA_VISIBLE_DEVICES=1 bash run_streamvln_eval.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
set -a
source "${NAV_ROOT}/config/r2r_ce_sota_reproduction.env"
set +a

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MAX_EPISODES="${MAX_EPISODES:--1}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 1000))}"
TAG="full"
if [[ "${MAX_EPISODES}" != "-1" && "${MAX_EPISODES}" -gt 0 ]]; then
  TAG="smoke${MAX_EPISODES}"
fi
OUT="${RESULT_ROOT}/streamvln/${TAG}_${RUN_ID}"
LOG_DIR="${LOG_ROOT}/streamvln"
mkdir -p "${OUT}/metrics" "${OUT}/logs" "${LOG_DIR}"
LOG="${LOG_DIR}/eval_${TAG}_${RUN_ID}.log"

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PYTHONPATH="${STREAMVLN_ROOT}/streamvln:${STREAMVLN_ROOT}:${PYTHONPATH:-}"
export PATH="${STREAMVLN_VENV}/bin:${PATH}"

# Prefer absolute ckpt; keep symlink for relative name
ln -sfn "${STREAMVLN_CKPT_DIR}" \
  "${STREAMVLN_ROOT}/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
CKPT_ARG="${STREAMVLN_CKPT_DIR}"

{
  echo "run_id=${RUN_ID} tag=${TAG} max_episodes=${MAX_EPISODES}"
  echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
  "${STREAMVLN_VENV}/bin/python" -c 'import torch,habitat_sim,sys; print(sys.version); print(torch.__version__, torch.cuda.device_count()); print(habitat_sim.__version__)'
  cd "${STREAMVLN_ROOT}"
  "${STREAMVLN_VENV}/bin/torchrun" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    streamvln/streamvln_eval.py \
      --model_path "${CKPT_ARG}" \
      --eval_split "${VLN_SPLIT}" \
      --output_path "${OUT}/habitat_out" \
      --max_episodes "${MAX_EPISODES}"
} 2>&1 | tee "${LOG}"

cp -a "${LOG}" "${OUT}/logs/" || true
if [[ -f "${OUT}/habitat_out/result.json" ]]; then
  cp -a "${OUT}/habitat_out/result.json" "${OUT}/metrics/" || true
fi
cat > "${OUT}/metrics/summary.json" <<EOF
{
  "method": "StreamVLN",
  "run_id": "${RUN_ID}",
  "tag": "${TAG}",
  "split": "${VLN_SPLIT}",
  "max_episodes": ${MAX_EPISODES},
  "checkpoint": "${STREAMVLN_HF_REPO}",
  "source": "Reproduced by us",
  "floor_r2r_sr": ${STREAMVLN_FLOOR_R2R_SR},
  "log": "${LOG}",
  "output": "${OUT}"
}
EOF
echo "Results: ${OUT}"
echo "Log: ${LOG}"
