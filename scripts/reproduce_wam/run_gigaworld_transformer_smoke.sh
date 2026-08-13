#!/usr/bin/env bash
set -euo pipefail

BASE="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln"
REPO="${BASE}/giga-world-policy"
VENV="${BASE}/virtual_env/.venv_gigaworld_policy"
LOG_ROOT="${BASE}/NAV/log/gigaworld_policy"
RESULT_ROOT="${BASE}/NAV/result/gigaworld_policy/smoke_transformer"
SHARE_ROOT="/sharedata/GigaWorld-Policy"

GPU_ID="${GPU_ID:-0}"
STEPS="${STEPS:-2}"
ACTION_HORIZON="${ACTION_HORIZON:-48}"
LATENT_H="${LATENT_H:-24}"
LATENT_W="${LATENT_W:-20}"
CKPT="${CKPT:-${SHARE_ROOT}/Giga-World-Policy-0.5}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export CKPT RUN_ID STEPS ACTION_HORIZON LATENT_H LATENT_W RESULT_ROOT

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}/${RUN_ID}"
cd "${REPO}"
source "${VENV}/bin/activate"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HOME="${HF_HOME:-/sharedata/huggingface}"
export PYTHONPATH="${REPO}:${REPO}/third_party/giga-models:${REPO}/third_party/giga-train:${REPO}/third_party/giga-datasets:${PYTHONPATH:-}"
export GIGA_MODELS_LIGHT_IMPORT=1

if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "[GigaWorld] checkpoint config missing: ${CKPT}/config.json" >&2
  exit 2
fi
if ! ls "${CKPT}"/diffusion_pytorch_model*.safetensors >/dev/null 2>&1; then
  echo "[GigaWorld] checkpoint shards missing under: ${CKPT}" >&2
  exit 2
fi

python - <<'PY'
import json
import os
import time
from pathlib import Path

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from world_action_model.models import CasualWorldActionTransformer_MoT

base = Path("/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln")
run_id = os.environ["RUN_ID"]
out_dir = base / "NAV" / "result" / "gigaworld_policy" / "smoke_transformer" / run_id
out_dir.mkdir(parents=True, exist_ok=True)

ckpt = os.environ["CKPT"]
steps = int(os.environ["STEPS"])
action_horizon = int(os.environ["ACTION_HORIZON"])
latent_h = int(os.environ["LATENT_H"])
latent_w = int(os.environ["LATENT_W"])
device = "cuda"
dtype = torch.bfloat16

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()
model = CasualWorldActionTransformer_MoT.from_pretrained(ckpt, torch_dtype=dtype).to(device)
model.eval()
model._enable_action_only_prefix_cache = True
torch.cuda.synchronize()
t_load = time.perf_counter() - t0

cfg = model.config
ref_latents = torch.randn(1, int(cfg.in_channels), 1, latent_h, latent_w, device=device, dtype=dtype)
empty_future = ref_latents[:, :, 1:1]
state = torch.zeros(1, 1, int(cfg.in_action_channels), device=device, dtype=dtype)
action = torch.randn(1, action_horizon, int(cfg.in_action_channels), device=device, dtype=dtype)
encoder_hidden_states = torch.zeros(1, 64, int(cfg.text_dim), device=device, dtype=dtype)
scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
scheduler.set_timesteps(steps, device=device)

num_ref_tokens = (latent_h // int(cfg.patch_size[1])) * (latent_w // int(cfg.patch_size[2]))
total_tokens = state.shape[1] + num_ref_tokens + action.shape[1]

times = []
with torch.inference_mode():
    model.reset_action_only_prefix_cache()
    for t in scheduler.timesteps:
        timestep = torch.zeros(1, total_tokens, device=device, dtype=dtype)
        timestep[:, state.shape[1] + num_ref_tokens:] = t.to(dtype)
        torch.cuda.synchronize()
        ts = time.perf_counter()
        pred = model(
            ref_latents=ref_latents,
            noisy_latents=empty_future,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            action=action,
            state=state,
            action_only=True,
        )
        action = scheduler.step(pred, t, action, return_dict=False)[0]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - ts)

peak_gib = torch.cuda.max_memory_allocated() / 1024**3
summary = {
    "status": "ok",
    "repo_commit": os.popen("git -C /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/giga-world-policy rev-parse --short HEAD").read().strip(),
    "checkpoint": ckpt,
    "device": torch.cuda.get_device_name(0),
    "num_inference_steps": steps,
    "action_horizon": action_horizon,
    "ref_latent_shape": list(ref_latents.shape),
    "load_seconds": t_load,
    "step_seconds": times,
    "infer_seconds": sum(times),
    "seconds_per_denoise_step": sum(times) / max(len(times), 1),
    "peak_allocated_gib": peak_gib,
    "action_shape": list(action.shape),
    "action_expert_dim": int(cfg.action_expert_dim),
    "num_layers": int(cfg.num_layers),
}
(out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
