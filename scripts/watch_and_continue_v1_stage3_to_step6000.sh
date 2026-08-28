#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?"usage: $0 <h4_natural|h1_balanced> <wait_pid> [gpu_index] [target_step]"}
WAIT_PID=${2:?"missing pid of the currently running training process"}
GPU_INDEX=${3:-0}
TARGET_STEP=${4:-6000}

NAV_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
INF_WORLD_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World
PYTHON_BIN=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python
OUTPUT_ROOT="$NAV_ROOT/log/v1_stage3_r2r_single_action"

case "$MODE" in
  h4_natural)
    SOURCE_RUN="$OUTPUT_ROOT/stage3_r2r_fullhistory_natural_a4_mb1_ebs16_from_stage2step3400_2k_20260828"
    ACTION_CHUNK=4
    R2R_LOADER=stage3_r2r_full_history_natural_action_chunk
    TENSORBOARD_PORT=6042
    ;;
  h1_balanced)
    SOURCE_RUN="$OUTPUT_ROOT/stage3_r2r_single_action_balanced_resume1400_to2000_gpu1_20260828"
    ACTION_CHUNK=1
    R2R_LOADER=stage3_r2r_balanced_single_action
    TENSORBOARD_PORT=6043
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

LATEST_CHECKPOINT=$(find "$SOURCE_RUN/checkpoints" -maxdepth 1 -type f -name 'step_*.pt' -print | sort | tail -n 1)
if [[ -z "$LATEST_CHECKPOINT" ]]; then
  echo "no checkpoint found under $SOURCE_RUN/checkpoints" >&2
  exit 3
fi

CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
SOURCE_STEP=${CHECKPOINT_NAME#step_}
SOURCE_STEP=${SOURCE_STEP%.pt}
SOURCE_STEP=$((10#$SOURCE_STEP))
if (( SOURCE_STEP >= TARGET_STEP )); then
  echo "$MODE already reached step $SOURCE_STEP (target $TARGET_STEP)"
  exit 0
fi

REMAINING_STEPS=$((TARGET_STEP - SOURCE_STEP))
RUN_NAME="stage3_${MODE}_resume${SOURCE_STEP}_to${TARGET_STEP}_gpu${GPU_INDEX}_$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="$OUTPUT_ROOT/${RUN_NAME}.launch.log"

cd "$NAV_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export NAV_INF_WORLD_ROOT="$INF_WORLD_ROOT"
export PYTHONPATH="$NAV_ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" scripts/train_v1_stage3_r2r_single_action.py \
  --run-name "$RUN_NAME" \
  --device cuda:0 \
  --steps "$REMAINING_STEPS" \
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
  --seed 20260828 \
  --checkpoint "$LATEST_CHECKPOINT" \
  --action-chunk "$ACTION_CHUNK" \
  --r2r-loader "$R2R_LOADER" \
  --resume-optimizer \
  --step-offset "$SOURCE_STEP" \
  --tensorboard-port "$TENSORBOARD_PORT" \
  2>&1 | tee "$LAUNCH_LOG"
