#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
PYTHON="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python"
VARIANT="${1:?用法: run_spatialvid_short_training.sh latent_prefix|dit_condition GPU RUN_NAME}"
GPU="${2:-1}"
RUN_NAME="${3:?缺少 RUN_NAME}"

case "$VARIANT" in
  latent_prefix)
    RESUME="$NAV_ROOT/log/re10k-all-a-full-ebs4-from-infinite/full-final.pt"
    ;;
  dit_condition)
    RESUME="$NAV_ROOT/log/re10k-all-b-full-ebs4-from-infinite/full-final.pt"
    ;;
  *)
    echo "未知 variant: $VARIANT" >&2
    exit 2
    ;;
esac

test -s "$RESUME"

exec "$PYTHON" "$NAV_ROOT/scripts/train_infinite_register.py" \
  --variant "$VARIANT" \
  --latent-cache /sharedata/NAV/derived/latents/spatialvid \
  --action-cache /sharedata/NAV/derived/actions/spatialvid_short \
  --resume-checkpoint "$RESUME" \
  --train-scope full \
  --device "cuda:$GPU" \
  --run-name "$RUN_NAME" \
  --steps 20000 \
  --save-every 500 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --min-chunks 2 \
  --max-chunks 3 \
  --shuffle-seed 20260728 \
  --plateau-window 200 \
  --plateau-patience 3 \
  --plateau-min-relative-improvement 0.01 \
  --min-steps-before-plateau 2000
