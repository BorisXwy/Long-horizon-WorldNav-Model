#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="${PYTHON:-python}"
DEVICE="${1:-cpu}"

export PYTHONPATH="${NAV_ROOT}/src:${PYTHONPATH:-}"

exec "${PYTHON}" "${NAV_ROOT}/scripts/smoke_v1_full_pipeline.py" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --history-steps "${HISTORY_STEPS:-3}" \
  --hidden-dim "${HIDDEN_DIM:-64}" \
  --register-tokens "${REGISTER_TOKENS:-8}" \
  --layers "${LAYERS:-2}" \
  --heads "${HEADS:-4}" \
  --latent-t "${LATENT_T:-2}" \
  --latent-h "${LATENT_H:-8}" \
  --latent-w "${LATENT_W:-8}" \
  --text-tokens "${TEXT_TOKENS:-3}"
