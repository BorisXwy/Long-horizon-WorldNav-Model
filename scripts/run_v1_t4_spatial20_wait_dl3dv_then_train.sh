#!/usr/bin/env bash
set -euo pipefail

# 等 V1 T4 spatial20 中的 DL3DV latent 全部完成后：
# 1. 停止当前双卡 latent workers；
# 2. 用 GPU0 继续补 spatial20 latent；
# 3. 用 GPU1 启动 Stage One T4 long-history 训练。

NAV_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
VENV=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world
PY="${VENV}/bin/python"

ENCODE_SCRIPT="${NAV_ROOT}/scripts/datasets/encode_v1_t4_micro_latents.py"
TRAIN_SCRIPT="${NAV_ROOT}/scripts/train_v1_wan_stage1_t4_history.py"

MANIFEST=/sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes_spatial20.jsonl
LATENT_ROOT=/sharedata/NAV/derived/v1/t4_micro_latents_spatial20
DL3DV_TARGET=141

LOG_ROOT="${NAV_ROOT}/log/v1_data_prep/t4_micro"
WATCH_RUN="${LOG_ROOT}/wait_dl3dv_then_train_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${WATCH_RUN}"
ln -sfn "${WATCH_RUN}" "${LOG_ROOT}/latest_wait_dl3dv_then_train"

echo "[watcher] start $(date '+%F %T')" | tee -a "${WATCH_RUN}/watcher.log"
echo "[watcher] manifest=${MANIFEST}" | tee -a "${WATCH_RUN}/watcher.log"
echo "[watcher] latent_root=${LATENT_ROOT}" | tee -a "${WATCH_RUN}/watcher.log"

while true; do
  done_count=$(find "${LATENT_ROOT}/dl3dv" -name '*.pt' 2>/dev/null | wc -l | awk '{print $1}')
  err_count=0
  if [ -f "${LOG_ROOT}/current_spatial20_run_dir.txt" ]; then
    run_dir=$(cat "${LOG_ROOT}/current_spatial20_run_dir.txt")
    if [ -d "${run_dir}" ]; then
      err_count=$( (grep -R "OutOfMemory\\|CUDA out of memory\\|Traceback" -n "${run_dir}" 2>/dev/null || true) | wc -l | awk '{print $1}' )
    fi
  fi
  echo "[watcher] $(date '+%F %T') dl3dv=${done_count}/${DL3DV_TARGET} errors=${err_count}" | tee -a "${WATCH_RUN}/watcher.log"
  if [ "${done_count}" -ge "${DL3DV_TARGET}" ]; then
    break
  fi
  sleep 300
done

echo "[watcher] DL3DV ready; switching resources $(date '+%F %T')" | tee -a "${WATCH_RUN}/watcher.log"

# 停掉当前 12 路双卡 latent worker。
for session in $(tmux ls 2>/dev/null | awk -F: '/^nav_v1_t4_latent_worker[0-9]+:/ {print $1}'); do
  tmux kill-session -t "${session}" || true
done

sleep 5

# GPU0 用 6 路覆盖完整 manifest，依靠 sidecar skip 已完成 RE10K/DL3DV/partial SpatialVID。
CONTINUE_RUN="${LOG_ROOT}/spatial20_gpu0_continue_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${CONTINUE_RUN}"
ln -sfn "${CONTINUE_RUN}" "${LOG_ROOT}/latest_spatial20_gpu0_continue"
echo "${CONTINUE_RUN}" > "${LOG_ROOT}/current_spatial20_gpu0_continue_run_dir.txt"

for i in $(seq 0 5); do
  tmux new-session -d -s "nav_v1_t4_latent_spatial20_gpu0_${i}" \
    "bash -lc 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ${PY} ${ENCODE_SCRIPT} --manifest ${MANIFEST} --output ${LATENT_ROOT} --device cuda:0 --num-shards 6 --shard-index ${i} 2>&1 | tee -a ${CONTINUE_RUN}/worker${i}.log'"
done
echo "[watcher] restarted latent continuation on GPU0: ${CONTINUE_RUN}" | tee -a "${WATCH_RUN}/watcher.log"

# GPU1 启动第一轮 DL3DV long-history full finetune。
TRAIN_NAME="v1-wan-stage1-t4-window-longhist-mb1-ebs16-1000-$(date +%Y%m%d-%H%M%S)"
TRAIN_LOG="${NAV_ROOT}/log/${TRAIN_NAME}.tmux.log"
tmux new-session -d -s nav_v1_t4_stage1_dl3dv_longhist \
  "bash -lc 'source ${VENV}/bin/activate; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ${PY} ${TRAIN_SCRIPT} --data-root ${LATENT_ROOT} --device cuda:1 --run-name ${TRAIN_NAME} --variant latent_prefix --train-scope full --history-iw-chunks 1,4,8,16 --batch-size 1 --gradient-accumulation-steps 16 --steps 1000 --save-every 200 --log-every 10 2>&1 | tee -a ${TRAIN_LOG}'"

echo "[watcher] started train session=nav_v1_t4_stage1_dl3dv_longhist run=${NAV_ROOT}/log/${TRAIN_NAME}" | tee -a "${WATCH_RUN}/watcher.log"
echo "[watcher] complete $(date '+%F %T')" | tee -a "${WATCH_RUN}/watcher.log"
