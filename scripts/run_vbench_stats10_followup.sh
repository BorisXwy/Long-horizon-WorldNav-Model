#!/usr/bin/env bash
# 在已启动 A/B 生成后，自动生成 InfiniteWorld、评测三版并汇总置信区间。
set -euo pipefail

REPO_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln"
NAV_ROOT="$REPO_ROOT/NAV"
INFINITE_ROOT="$REPO_ROOT/Infinite-World"
VBENCH_ROOT="$REPO_ROOT/VBench"
INFINITE_ENV="$REPO_ROOT/virtual_env/.venv_infinite_world"
VBENCH_ENV="$REPO_ROOT/virtual_env/.venv_vbench"

RESULT_ROOT="$NAV_ROOT/result/vbench"
A_ROOT="$RESULT_ROOT/nav_a_stats10"
B_ROOT="$RESULT_ROOT/nav_b_stats10"
I_ROOT="$RESULT_ROOT/infiniteworld_stats10"
SUMMARY_ROOT="$RESULT_ROOT/threeway_stats10"
A_VIDEOS="$A_ROOT/videos"
B_VIDEOS="$B_ROOT/videos"
I_VIDEOS="$I_ROOT/videos"
A_RESULTS="$A_ROOT/metrics"
B_RESULTS="$B_ROOT/metrics"
I_RESULTS="$I_ROOT/metrics"

video_count() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
}

# 等 A/B 均完成后复用 GPU 1，避免与 Register 推理争抢显存。
while [[ "$(video_count "$B_VIDEOS")" -lt 10 ]]; do
  sleep 30
done
while [[ "$(video_count "$A_VIDEOS")" -lt 10 ]]; do
  sleep 30
done
sleep 30

mkdir -p "$I_VIDEOS" "$I_RESULTS" "$I_ROOT/logs" \
  "$A_RESULTS" "$A_ROOT/logs" "$B_RESULTS" "$B_ROOT/logs" \
  "$SUMMARY_ROOT/metrics" "$SUMMARY_ROOT/logs"
cd "$INFINITE_ROOT"
export PATH="$INFINITE_ENV/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
export INFWORLD_MAX_PROMPTS=2
export INFWORLD_NUM_CHUNKS=2
export INFWORLD_SAMPLING_STEPS=30
export INFWORLD_CFG_SCALE=5.0
export INFWORLD_OUTPUT_DIR="$I_VIDEOS"
for seed in 40 41 42 43 44; do
  export INFWORLD_SEED="$seed"
  export INFWORLD_FILENAME_SUFFIX="_seed$(printf '%04d' "$seed")"
  python scripts/infworld_inference.py
done 2>&1 | tee "$I_ROOT/logs/inference.log"

cd "$VBENCH_ROOT"
source config/sharedata.env
export PATH="$VBENCH_ENV/bin:$PATH"
for spec in \
  "$I_VIDEOS|$I_RESULTS|$I_ROOT/logs/evaluation.log" \
  "$A_VIDEOS|$A_RESULTS|$A_ROOT/logs/evaluation.log" \
  "$B_VIDEOS|$B_RESULTS|$B_ROOT/logs/evaluation.log"; do
  IFS='|' read -r videos results evaluation_log <<< "$spec"
  CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --videos_path "$videos" \
    --dimension motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
    --mode custom_input \
    --load_ckpt_from_local True \
    --output_path "$results" \
    2>&1 | tee "$evaluation_log"
done

export PATH="$INFINITE_ENV/bin:$PATH"
python "$NAV_ROOT/scripts/summarize_vbench_threeway.py" \
  --infinite "$I_RESULTS" \
  --a "$A_RESULTS" \
  --b "$B_RESULTS" \
  --output-json "$SUMMARY_ROOT/metrics/summary.json" \
  --output-md "$SUMMARY_ROOT/metrics/report.md"
