#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
VENV=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world
PYTHON="${VENV}/bin/python"

DEVICE="${1:-cuda:0}"
RUN_NAME="${2:-v1-wan-stage1-t4-iw-$(date +%Y%m%d-%H%M%S)}"
HISTORY_IW_CHUNKS="${3:-1,4,8,16}"

exec "${PYTHON}" "${NAV_ROOT}/scripts/train_v1_wan_stage1_t4_history.py" \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --history-iw-chunks "${HISTORY_IW_CHUNKS}" \
  --data-root /sharedata/NAV/derived/v1/t4_micro_latents_spatial20 \
  --checkpoint /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --variant latent_prefix \
  --train-scope full \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --steps 1000 \
  --save-every 200 \
  --log-every 10
