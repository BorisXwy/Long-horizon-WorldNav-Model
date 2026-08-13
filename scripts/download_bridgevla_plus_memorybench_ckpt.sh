#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/BridgeVLA"
CONDA_BASE="${CONDA_BASE:-/sharedata/anaconda3}"
ENV_NAME="${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PATH="/mnt/pool1/sharehome/xiewenyuan/.conda/envs/${ENV_NAME}/bin:${PATH}"
hash -r

export HF_HOME="${HF_HOME:-/sharedata/BridgeVLA/cache/hf}"
export BRIDGEVLA_CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-/sharedata/BridgeVLA/data/bridgevla_ckpt}"
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-/sharedata/BridgeVLA/data/bridgevla_data}"

cd "${REPO_ROOT}"
bash scripts/download_checkpoints_ms.sh --dest /sharedata/BridgeVLA/data \
  memorybench paligemma clip memorybench_cache
