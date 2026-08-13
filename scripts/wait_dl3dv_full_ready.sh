#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV"
LATENTS="/sharedata/NAV/derived/latents/full_episodes_v1/dl3dv"
READY="/sharedata/NAV/derived/stage_ready/full_episodes_v1_dl3dv.ready"
STATUS="$NAV_ROOT/log/data_prep/full-episodes-v1-20260729/status.log"
EXPECTED=141

while true; do
  COUNT="$(find "$LATENTS" -maxdepth 1 -type f -name '*.pt' \
    2>/dev/null | wc -l)"
  if ((COUNT >= EXPECTED)); then
    touch "$READY"
    echo "[$(date -Is)] DL3DV完整latent已就绪：$COUNT/$EXPECTED" \
      >>"$STATUS"
    exit 0
  fi
  sleep 60
done
