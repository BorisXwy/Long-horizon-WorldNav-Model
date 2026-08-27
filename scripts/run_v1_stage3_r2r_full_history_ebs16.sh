#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
INF_WORLD_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World
PYTHON_BIN=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python
RUN_NAME=${1:-stage3_r2r_fullhistory_natural_a4_mb1_ebs16_from_stage2step3400_2k_20260828}
GPU_INDEX=${GPU_INDEX:-0}
TENSORBOARD_PORT=${TENSORBOARD_PORT:-6040}

cd "$NAV_ROOT"
export NAV_INF_WORLD_ROOT="$INF_WORLD_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$NAV_ROOT/src"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" scripts/train_v1_stage3_r2r_single_action.py \
  --run-name "$RUN_NAME" \
  --device cuda:0 \
  --steps 2000 \
  --batch-size 1 \
  --grad-accum 16 \
  --backbone-lr 2e-6 \
  --policy-lr 1e-4 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --lambda-video-replay 0.25 \
  --lambda-pose-replay 0.05 \
  --dtype bf16 \
  --save-every 200 \
  --log-every 1 \
  --seed 20260826 \
  --checkpoint log/v1_stage2_final_cotrain/stage2_from_step3000_continue_lr2e6_1k_20260825/checkpoints/step_003400.pt \
  --action-chunk 4 \
  --r2r-loader stage3_r2r_full_history_natural_action_chunk \
  --tensorboard-port "$TENSORBOARD_PORT"
