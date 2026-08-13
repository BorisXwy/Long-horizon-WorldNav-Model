#!/usr/bin/env bash
set -euo pipefail

BASE="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln"
REPO="${BASE}/FastWAM"
VENV="${BASE}/virtual_env/.venv_fastwam"
LOG_ROOT="${BASE}/NAV/log/fastwam"
RESULT_ROOT="${BASE}/NAV/result/fastwam/smoke"
SHARE_ROOT="/sharedata/FastWAM"

GPU_ID="${GPU_ID:-0}"
STEPS="${STEPS:-2}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
HEIGHT="${HEIGHT:-224}"
WIDTH="${WIDTH:-448}"
CKPT="${CKPT:-${SHARE_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
ACTION_DIT="${ACTION_DIT:-${SHARE_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export CKPT ACTION_DIT RUN_ID STEPS ACTION_HORIZON HEIGHT WIDTH RESULT_ROOT

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}/${RUN_ID}" "${SHARE_ROOT}/checkpoints"
cd "${REPO}"
source "${VENV}/bin/activate"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export DIFFSYNTH_MODEL_BASE_PATH="${SHARE_ROOT}/models"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export HF_HOME="${HF_HOME:-/sharedata/huggingface}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "${ACTION_DIT}" ]]; then
  echo "[FastWAM] ActionDiT backbone not found, preprocessing: ${ACTION_DIT}"
  TMP_MODEL_CONFIG="${LOG_ROOT}/fastwam_no_redirect_${RUN_ID}.yaml"
  export TMP_MODEL_CONFIG
  python - <<'PY'
import os
from pathlib import Path

src = Path("configs/model/fastwam.yaml")
dst = Path(os.environ["TMP_MODEL_CONFIG"])
text = src.read_text(encoding="utf-8")
if "redirect_common_files:" in text:
    text = text.replace("redirect_common_files: true", "redirect_common_files: false")
else:
    text += "\nredirect_common_files: false\n"
dst.write_text(text, encoding="utf-8")
print(f"[FastWAM] temporary local-only model config: {dst}")
PY
  python scripts/preprocess_action_dit_backbone.py \
    --model-config "${TMP_MODEL_CONFIG}" \
    --output "${ACTION_DIT}" \
    --device cuda \
    --dtype bfloat16
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[FastWAM] checkpoint missing: ${CKPT}" >&2
  exit 2
fi

python - <<'PY'
import json
import os
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

base = Path("/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln")
repo = base / "FastWAM"
result_dir = Path(os.environ.get("RESULT_ROOT", "")) if os.environ.get("RESULT_ROOT") else None
run_id = os.environ["RUN_ID"]
out_dir = base / "NAV" / "result" / "fastwam" / "smoke" / run_id
out_dir.mkdir(parents=True, exist_ok=True)

ckpt = os.environ["CKPT"]
action_dit = os.environ["ACTION_DIT"]
steps = int(os.environ["STEPS"])
action_horizon = int(os.environ["ACTION_HORIZON"])
height = int(os.environ["HEIGHT"])
width = int(os.environ["WIDTH"])

with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
    cfg = compose(
        config_name="train",
        overrides=[
            "task=libero_uncond_2cam224_1e-4",
            f"model.action_dit_pretrained_path={action_dit}",
            "model.redirect_common_files=false",
            "model.load_text_encoder=false",
            "model.mot_checkpoint_mixed_attn=false",
        ],
    )

device = "cuda"
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()
model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
load_info = model.load_checkpoint(ckpt)
model.eval()
torch.cuda.synchronize()
t_load = time.perf_counter() - t0

context = torch.zeros(1, int(cfg.data.train.context_len), int(cfg.model.video_dit_config.text_dim), dtype=torch.bfloat16, device=device)
context_mask = torch.ones(1, int(cfg.data.train.context_len), dtype=torch.bool, device=device)
image = torch.rand(1, 3, height, width, dtype=torch.bfloat16, device=device) * 2 - 1
proprio_dim = cfg.model.get("proprio_dim")
proprio = torch.zeros(1, int(proprio_dim), dtype=torch.bfloat16, device=device) if proprio_dim is not None else None

torch.cuda.synchronize()
t1 = time.perf_counter()
pred = model.infer_action(
    prompt=None,
    context=context,
    context_mask=context_mask,
    input_image=image,
    action_horizon=action_horizon,
    proprio=proprio,
    num_inference_steps=steps,
    seed=123,
    rand_device="cpu",
)
torch.cuda.synchronize()
t_infer = time.perf_counter() - t1
peak_gib = torch.cuda.max_memory_allocated() / 1024**3

summary = {
    "status": "ok",
    "repo_commit": os.popen("git -C /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/FastWAM rev-parse --short HEAD").read().strip(),
    "checkpoint": ckpt,
    "action_dit": action_dit,
    "device": torch.cuda.get_device_name(0),
    "height": height,
    "width": width,
    "action_horizon": action_horizon,
    "num_inference_steps": steps,
    "load_seconds": t_load,
    "infer_seconds": t_infer,
    "seconds_per_denoise_step": t_infer / max(steps, 1),
    "peak_allocated_gib": peak_gib,
    "action_shape": list(pred["action"].shape),
    "checkpoint_step": load_info.get("step") if isinstance(load_info, dict) else None,
}
(out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
