#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world"
export PYTHONPATH="${NAV_ROOT}/src:/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World:${PYTHONPATH:-}"

variant="${1:-latent_prefix}"
steps="${2:-20}"
exec "${ENV_ROOT}/bin/python" "${NAV_ROOT}/scripts/train_register_smoke.py" \
  --variant "${variant}" --steps "${steps}"
