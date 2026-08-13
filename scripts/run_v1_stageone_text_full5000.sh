#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
VENV=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world
PYTHON="${VENV}/bin/python"

DEVICE="${1:-cuda:1}"
RUN_NAME="${2:-v1-stageone-text-fullmix-wan21official-lp-mb1-ebs16-5000-$(date +%Y%m%d-%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-/sharedata/NAV/derived/v1/t4_micro_latents_spatial20}"
TEXT_CACHE_ROOT="${TEXT_CACHE_ROOT:-/sharedata/NAV/derived/v1/text_embeddings/t4_micro}"
DATASET_WEIGHTS="${DATASET_WEIGHTS:-spatialvid=0.45,dl3dv=0.35,re10k=0.15,argoverse2=0.05}"

exec "${PYTHON}" "${NAV_ROOT}/scripts/train_v1_wan_stage1_t4_history.py" \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --history-iw-chunks 1,4,8,16 \
  --data-root "${DATA_ROOT}" \
  --checkpoint /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --text-embedding /sharedata/RealEstate10K/nav_register/text/empty_umt5.pt \
  --text-cache-root "${TEXT_CACHE_ROOT}" \
  --dataset-weights "${DATASET_WEIGHTS}" \
  --variant latent_prefix \
  --train-scope full \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --steps 5000 \
  --save-every 1000 \
  --log-every 10
