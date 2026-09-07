#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python"
CONFIG="${1:-${NAV_ROOT}/config/giga_nav/giga_nav_wan21_h8_cotrain.yaml}"
shift $(( $# > 0 ? 1 : 0 ))

export PYTHONPATH="${NAV_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
exec "${PYTHON}" "${NAV_ROOT}/scripts/train_giga_nav_multitask.py" --config "${CONFIG}" "$@"
