#!/usr/bin/env bash
# Download open checkpoints for ≥StreamVLN baselines (NAV-EVL-002).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
set -a
source "${NAV_ROOT}/config/r2r_ce_sota_reproduction.env"
set +a

mkdir -p "${BASELINE_CKPT_ROOT}" "${LOG_ROOT}"
LOG="${LOG_ROOT}/download_ckpts_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

HF_BIN=""
if command -v hf >/dev/null 2>&1; then
  HF_BIN=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN=huggingface-cli
else
  # prefer streamvln / data_prep venv
  for c in \
    "${STREAMVLN_VENV}/bin/hf" \
    "${VENV_ROOT}/.venv_data_prep/bin/hf" \
    "${VENV_ROOT}/.venv_data_prep/bin/huggingface-cli"; do
    if [[ -x "${c}" ]]; then HF_BIN="${c}"; break; fi
  done
fi

if [[ -z "${HF_BIN}" ]]; then
  echo "[info] installing huggingface_hub into streamvln venv for downloads"
  "${STREAMVLN_VENV}/bin/pip" install -U 'huggingface_hub[cli]'
  HF_BIN="${STREAMVLN_VENV}/bin/hf"
fi

download_repo() {
  local repo="$1"
  local dest="$2"
  mkdir -p "${dest}"
  echo "[download] ${repo} -> ${dest}"
  if [[ "${HF_BIN}" == *huggingface-cli ]]; then
    "${HF_BIN}" download "${repo}" --local-dir "${dest}" --resume-download
  else
    "${HF_BIN}" download "${repo}" --local-dir "${dest}"
  fi
}

# P0 StreamVLN
download_repo "${STREAMVLN_HF_REPO}" "${STREAMVLN_CKPT_DIR}"
# Also expose under StreamVLN working dir name expected by eval script
ln -sfn "${STREAMVLN_CKPT_DIR}" \
  "${STREAMVLN_ROOT}/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln" || true

# P1 DualVLN
download_repo "${DUALVLN_HF_REPO}" "${DUALVLN_CKPT_DIR}"

# P2 optional NavDP dual
if [[ "${DOWNLOAD_NAVDP:-0}" == "1" ]]; then
  download_repo "${NAVDP_HF_REPO}" "${NAVDP_CKPT_DIR}"
fi

echo "Checkpoints under ${BASELINE_CKPT_ROOT}"
du -sh "${BASELINE_CKPT_ROOT}"/* 2>/dev/null || true
echo "Log: ${LOG}"
