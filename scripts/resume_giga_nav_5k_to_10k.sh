#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_DEVICE="${GPU_DEVICE:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-48000}"
STABLE_CHECKS_REQUIRED="${STABLE_CHECKS_REQUIRED:-8}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$ROOT/log/giga_nav_wan21_h8_ebs32_20260903_003600/step_005000.pt}"
RUN_DIR="${RUN_DIR:-$ROOT/log/giga_nav_wan21_h8_ebs32_resume5000_to10000_20260906}"

stable_checks=0
while (( stable_checks < STABLE_CHECKS_REQUIRED )); do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_DEVICE" | tr -d ' ')"
    if pgrep -f "evaluate_[p]oint_navigation.py.*--device cuda:${GPU_DEVICE}" >/dev/null \
        || pgrep -f "habitat_[p]oint_navigation.py.*--device cuda:${GPU_DEVICE}" >/dev/null \
        || (( free_mib < MIN_FREE_MIB )); then
        stable_checks=0
        echo "[$(date '+%F %T')] waiting for cuda:${GPU_DEVICE}: free=${free_mib} MiB"
    else
        stable_checks=$((stable_checks + 1))
        echo "[$(date '+%F %T')] cuda:${GPU_DEVICE} free=${free_mib} MiB; stable=${stable_checks}/${STABLE_CHECKS_REQUIRED}"
    fi
    if (( stable_checks < STABLE_CHECKS_REQUIRED )); then
        sleep "$CHECK_INTERVAL_SECONDS"
    fi
done

mkdir -p "$RUN_DIR"
echo "[$(date '+%F %T')] starting GigaNav optimizer-state resume: step 5000 -> 10000"
exec bash "$ROOT/scripts/run_giga_nav_train.sh" \
    --device "cuda:${GPU_DEVICE}" \
    --batch-size 1 \
    --action-horizon 8 \
    --grad-accumulation-steps 32 \
    --lr 6e-5 \
    --steps 10000 \
    --save-interval 1000 \
    --resume "$SOURCE_CHECKPOINT" \
    --output "$RUN_DIR"
