#!/usr/bin/env bash
# 在 GPU 0 与 VGGT 共存，串行完成 A/B/InfiniteWorld 的 20 样本生成与 VBench。
set -euo pipefail

REPO_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln"
NAV_ROOT="$REPO_ROOT/NAV"
INFINITE_ROOT="$REPO_ROOT/Infinite-World"
VBENCH_ROOT="$REPO_ROOT/VBench"
INFINITE_ENV="$REPO_ROOT/virtual_env/.venv_infinite_world"
VBENCH_ENV="$REPO_ROOT/virtual_env/.venv_vbench"
RESULT_ROOT="$NAV_ROOT/result/vbench"
RUN_NAME="threeway_stats20_gpu0"
RUN_ROOT="$RESULT_ROOT/$RUN_NAME"
A_ROOT="$RESULT_ROOT/nav_a_stats20_gpu0"
B_ROOT="$RESULT_ROOT/nav_b_stats20_gpu0"
I_ROOT="$RESULT_ROOT/infiniteworld_stats20_gpu0"
A_CKPT_SOURCE="$NAV_ROOT/log/re10k-all-a-full-ebs4-from-infinite/full-step-001000.pt"
B_CKPT_SOURCE="$NAV_ROOT/log/re10k-all-b-full-ebs4-from-infinite/full-step-001000.pt"
LOCAL_CKPT_ROOT="/tmp/xiewenyuan_nav_vbench_ckpts"
A_CKPT="$LOCAL_CKPT_ROOT/a-full-step-001000.pt"
B_CKPT="$LOCAL_CKPT_ROOT/b-full-step-001000.pt"
SEEDS=(100 101 102 103 104 105 106 107 108 109)

mkdir -p \
  "$RUN_ROOT/logs" "$RUN_ROOT/metrics" \
  "$A_ROOT/videos" "$A_ROOT/metrics" "$A_ROOT/logs" \
  "$B_ROOT/videos" "$B_ROOT/metrics" "$B_ROOT/logs" \
  "$I_ROOT/videos" "$I_ROOT/metrics" "$I_ROOT/logs"

status() {
  printf '[%s] %s\n' "$(date -Is)" "$1" | tee -a "$RUN_ROOT/logs/pipeline.log"
}

stage_checkpoint() {
  local source="$1"
  local target="$2"
  if [[ ! -f "$target" ]] || \
     [[ "$(stat -c %s "$target")" != "$(stat -c %s "$source")" ]]; then
    status "stage=checkpoint_copy source=$source target=$target"
    cp --reflink=auto "$source" "$target.tmp"
    mv "$target.tmp" "$target"
  fi
}

mkdir -p "$LOCAL_CKPT_ROOT"
stage_checkpoint "$A_CKPT_SOURCE" "$A_CKPT"
stage_checkpoint "$B_CKPT_SOURCE" "$B_CKPT"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="$INFINITE_ENV/bin:$PATH"

status "stage=A_generation samples=20 prompts=0,1 seeds=${SEEDS[*]}"
python "$NAV_ROOT/scripts/infer_register_vbench.py" \
  --variant latent_prefix \
  --checkpoint "$A_CKPT" \
  --output-dir "$A_ROOT/videos" \
  --device cuda:0 --chunks 2 --steps 30 --cfg-scale 5.0 \
  --prompt-indices 0 1 --seeds "${SEEDS[@]}" \
  2>&1 | tee -a "$A_ROOT/logs/inference.log"
touch "$A_ROOT/.generation_complete"

status "stage=B_generation samples=20"
python "$NAV_ROOT/scripts/infer_register_vbench.py" \
  --variant dit_condition \
  --checkpoint "$B_CKPT" \
  --output-dir "$B_ROOT/videos" \
  --device cuda:0 --chunks 2 --steps 30 --cfg-scale 5.0 \
  --prompt-indices 0 1 --seeds "${SEEDS[@]}" \
  2>&1 | tee -a "$B_ROOT/logs/inference.log"
touch "$B_ROOT/.generation_complete"

status "stage=InfiniteWorld_generation samples=20"
cd "$INFINITE_ROOT"
export INFWORLD_MAX_PROMPTS=2
export INFWORLD_NUM_CHUNKS=2
export INFWORLD_SAMPLING_STEPS=30
export INFWORLD_CFG_SCALE=5.0
export INFWORLD_OUTPUT_DIR="$I_ROOT/videos"
for seed in "${SEEDS[@]}"; do
  export INFWORLD_SEED="$seed"
  export INFWORLD_FILENAME_SUFFIX="_seed$(printf '%04d' "$seed")"
  python scripts/infworld_inference.py
done 2>&1 | tee -a "$I_ROOT/logs/inference.log"
touch "$I_ROOT/.generation_complete"

status "stage=VBench_evaluation dimensions=4"
cd "$VBENCH_ROOT"
source config/sharedata.env
export PATH="$VBENCH_ENV/bin:$PATH"
for spec in \
  "$A_ROOT/videos|$A_ROOT/metrics|$A_ROOT/logs/evaluation.log" \
  "$B_ROOT/videos|$B_ROOT/metrics|$B_ROOT/logs/evaluation.log" \
  "$I_ROOT/videos|$I_ROOT/metrics|$I_ROOT/logs/evaluation.log"; do
  IFS='|' read -r videos metrics evaluation_log <<< "$spec"
  python evaluate.py \
    --videos_path "$videos" \
    --dimension motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
    --mode custom_input \
    --load_ckpt_from_local True \
    --output_path "$metrics" \
    2>&1 | tee -a "$evaluation_log"
done

status "stage=paired_bootstrap_summary"
export PATH="$INFINITE_ENV/bin:$PATH"
python "$NAV_ROOT/scripts/summarize_vbench_threeway.py" \
  --infinite "$I_ROOT/metrics" \
  --a "$A_ROOT/metrics" \
  --b "$B_ROOT/metrics" \
  --expected-samples 20 \
  --output-json "$RUN_ROOT/metrics/summary.json" \
  --output-md "$RUN_ROOT/metrics/report.md"
touch "$RUN_ROOT/.pipeline_complete"
status "stage=complete"
