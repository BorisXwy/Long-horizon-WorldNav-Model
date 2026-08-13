#!/usr/bin/env bash
# 兼容已启动的旧 supervisor：若其在 latent 阶段失败，则自动以新版逻辑续跑。
set -euo pipefail

ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln"
SESSION="nav_full_pipeline"
PIPELINE_ID="${1:?需要 pipeline id}"

while true; do
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    exit 0
  fi
  dead="$(tmux display-message -pt "$SESSION:0" '#{pane_dead}')"
  if [[ "$dead" == "1" ]]; then
    exit_status="$(tmux display-message -pt "$SESSION:0" '#{pane_dead_status}')"
    if [[ "$exit_status" == "0" ]]; then
      exit 0
    fi
    tmux kill-session -t "$SESSION"
    tmux new-session -d -s "$SESSION" -c "$ROOT" \
      "bash NAV/scripts/run_full_background_pipeline.sh '$PIPELINE_ID'"
    tmux set-option -t "$SESSION" remain-on-exit on
    exit 0
  fi
  sleep 60
done
