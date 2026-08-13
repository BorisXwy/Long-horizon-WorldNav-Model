#!/usr/bin/env bash
# Bootstrap InternNav / DualVLN env at virtual_env/.venv_internnav (NAV-EVL-002).
# Folder is a conda prefix (`conda create -p`) so habitat-sim can be installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
set -a
source "${NAV_ROOT}/config/r2r_ce_sota_reproduction.env"
set +a

LOG_DIR="${LOG_ROOT}/internnav"
mkdir -p "${LOG_DIR}" "${BASELINE_CKPT_ROOT}"
LOG="${LOG_DIR}/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== setup InternNav env (conda-prefix @ venv path) ==="
echo "env: ${INTERNAV_VENV}"
echo "code: ${INTERNAV_ROOT}"

if [[ ! -f "${INTERNAV_VENV}/conda-meta/history" ]]; then
  if [[ -d "${INTERNAV_VENV}" ]]; then
    mv "${INTERNAV_VENV}" "${INTERNAV_VENV}.bak_nonconda_$(date +%Y%m%d_%H%M%S)"
  fi
  # habitat-sim 0.2.4 conda builds are py3.9-only
  conda create -y -p "${INTERNAV_VENV}" python=3.9 pip
fi

export PATH="${INTERNAV_VENV}/bin:${PATH}"
PY="${INTERNAV_VENV}/bin/python"
PIP="${INTERNAV_VENV}/bin/pip"

if [[ -f "${INTERNAV_ROOT}/.gitmodules" ]]; then
  git -C "${INTERNAV_ROOT}" submodule update --init --recursive || true
fi

if ! "${PY}" -c 'import habitat_sim' 2>/dev/null; then
  conda install -y -p "${INTERNAV_VENV}" habitat-sim=0.2.4 withbullet headless \
    -c conda-forge -c aihabitat
fi

HABLAB="${WM_ROOT}/habitat-lab"
if [[ ! -d "${HABLAB}/.git" ]]; then
  git clone --depth 1 --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git "${HABLAB}"
fi

"${PIP}" install -U pip setuptools wheel
if ! "${PY}" -c 'import torch' 2>/dev/null; then
  if ! "${PIP}" install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118; then
    "${PIP}" install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124
  fi
fi

"${PIP}" install 'numpy<2'
"${PIP}" install -e "${HABLAB}/habitat-lab" || true

cd "${INTERNAV_ROOT}"
"${PIP}" install -e '.[model]' --no-build-isolation || "${PIP}" install -e . --no-build-isolation
"${PIP}" install -e '.[habitat]' --no-build-isolation || echo "[warn] habitat extras partial"
"${PIP}" install 'numpy<2'
# DualVLN weights require Lumina FFN dim 1024; newer diffusers build 1536 → size mismatch
# (InternNav#322; inference_only_demo uses 0.31.0)
"${PIP}" install 'diffusers==0.31.0' \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn || \
  "${PIP}" install 'diffusers==0.31.0'

"${PY}" - <<'PY'
import numpy, torch, habitat_sim, internnav, sys
print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("habitat_sim", habitat_sim.__version__)
print("internnav", getattr(internnav, "__file__", "ok"))
print("OK")
PY

echo "Log: ${LOG}"
echo "Done setup_internnav_env.sh"
