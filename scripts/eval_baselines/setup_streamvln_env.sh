#!/usr/bin/env bash
# Bootstrap StreamVLN env at virtual_env/.venv_streamvln (NAV-EVL-002).
# Note: habitat-sim has no stable pip wheel; the folder is a conda prefix
# created with `conda create -p` (same path convention as project venvs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
set -a
source "${NAV_ROOT}/config/r2r_ce_sota_reproduction.env"
set +a

LOG_DIR="${LOG_ROOT}/streamvln"
mkdir -p "${LOG_DIR}" "${BASELINE_CKPT_ROOT}"
LOG="${LOG_DIR}/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== setup StreamVLN env (conda-prefix @ venv path) ==="
echo "env: ${STREAMVLN_VENV}"
echo "code: ${STREAMVLN_ROOT}"

if [[ ! -f "${STREAMVLN_VENV}/conda-meta/history" ]]; then
  if [[ -d "${STREAMVLN_VENV}" ]]; then
    mv "${STREAMVLN_VENV}" "${STREAMVLN_VENV}.bak_nonconda_$(date +%Y%m%d_%H%M%S)"
  fi
  conda create -y -p "${STREAMVLN_VENV}" python=3.9 pip
fi

export PATH="${STREAMVLN_VENV}/bin:${PATH}"
PY="${STREAMVLN_VENV}/bin/python"
PIP="${STREAMVLN_VENV}/bin/pip"

if ! "${PY}" -c 'import habitat_sim' 2>/dev/null; then
  conda install -y -p "${STREAMVLN_VENV}" habitat-sim=0.2.4 withbullet headless \
    -c conda-forge -c aihabitat
fi

HABLAB="${WM_ROOT}/habitat-lab"
if [[ ! -d "${HABLAB}/.git" ]]; then
  git clone --depth 1 --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git "${HABLAB}"
fi

"${PIP}" install -U pip setuptools wheel
"${PIP}" install -e "${HABLAB}/habitat-lab"
"${PIP}" install -e "${HABLAB}/habitat-baselines" || true

if ! "${PY}" -c 'import torch' 2>/dev/null; then
  "${PIP}" install 'torch==2.1.2' 'torchvision==0.16.2' 'torchaudio==2.1.2' \
    --index-url https://download.pytorch.org/whl/cu121
fi

"${PIP}" install 'numpy<2'
if ! "${PIP}" install -r "${STREAMVLN_ROOT}/requirements.txt"; then
  echo "[warn] full requirements failed; installing core subset"
  "${PIP}" install accelerate==0.28.0 einops transformers==4.40.0 peft \
    opencv-python-headless tqdm omegaconf imageio av decord sentencepiece 'numpy<2'
fi
"${PIP}" install 'numpy<2'

"${PY}" - <<'PY'
import numpy, torch, habitat_sim, habitat, sys
print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("habitat_sim", habitat_sim.__version__)
print("habitat", getattr(habitat, "__version__", "?"))
print("OK")
PY

echo "Log: ${LOG}"
echo "Done setup_streamvln_env.sh"
