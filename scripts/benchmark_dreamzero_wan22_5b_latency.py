#!/usr/bin/env python
"""DreamZero Wan2.2-TI2V-5B backbone latency probe.

本脚本只测 DreamZero 5B action head 中最核心的 Wan DiT / action-register
推理路径：synthetic latent + text context + CLIP context + noisy action/state。
它不依赖 DreamZero 训练 checkpoint 的 experiment_cfg，也不启动 websocket server。

默认不加载 Wan2.2 权重，适合快速测结构 latency；加 --load-weights 后会从
--wan22-dir 加载 base Wan2.2 DiT safetensors（action/state register 相关层仍为
DreamZero 随机初始化，因为 base Wan 没有这些层）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def _add_dreamzero_to_path(repo: Path) -> None:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _load_wan22_state(model: torch.nn.Module, wan22_dir: Path) -> dict:
    from safetensors.torch import load_file

    index_path = wan22_dir / "diffusion_pytorch_model.safetensors.index.json"
    single_path = wan22_dir / "diffusion_pytorch_model.safetensors"
    state = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(wan22_dir / shard)))
    elif single_path.exists():
        state = load_file(str(single_path))
    else:
        raise FileNotFoundError(f"未找到 Wan2.2 DiT safetensors: {wan22_dir}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "loaded_keys": len(state),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_key_examples": missing[:20],
        "unexpected_key_examples": unexpected[:20],
    }


def _make_kv_cache(model: torch.nn.Module, batch_size: int, dtype: torch.dtype, device: str):
    head_dim = model.dim // model.num_heads
    return [
        torch.zeros(
            [2, batch_size, 0, model.num_heads, head_dim],
            device=device,
            dtype=dtype,
        )
        for _ in range(model.num_layers)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamzero-repo", default="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/dreamzero")
    parser.add_argument("--wan22-dir", default="/sharedata/Wan2.2-TI2V-5B")
    parser.add_argument("--out", default="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/result/dreamzero_5b/wan22_5b_latency.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--dit-compute-steps",
        type=int,
        default=8,
        help="DreamZero 官方默认是 16 scheduler steps / 8 DiT compute steps；设为 16 表示每个 timestep 都跑 DiT。",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--load-weights", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.dit_compute_steps is not None:
        os.environ["NUM_DIT_STEPS"] = str(args.dit_compute_steps)

    _add_dreamzero_to_path(Path(args.dreamzero_repo))

    from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = args.device
    torch.cuda.set_device(torch.device(device))
    torch.backends.cuda.matmul.allow_tf32 = True

    model = CausalWanModel(
        model_type="ti2v",
        frame_seqlen=50,
        dim=3072,
        in_dim=48,
        out_dim=48,
        ffn_dim=14336,
        freq_dim=256,
        num_heads=24,
        num_layers=30,
        max_chunk_size=5,
        num_frame_per_block=2,
        num_action_per_block=24,
        num_state_per_block=1,
        action_dim=32,
        max_state_dim=64,
        hidden_size=64,
        concat_first_frame_latent=False,
    )
    load_report = None
    if args.load_weights:
        load_report = _load_wan22_state(model, Path(args.wan22_dir))
    model = model.to(device=device, dtype=dtype).eval()
    model.gradient_checkpointing = False

    b = args.batch_size
    latent_h, latent_w = 10, 20
    latent_f = 2
    seq_len = latent_f * 50
    x = torch.randn(b, 48, latent_f, latent_h, latent_w, device=device, dtype=dtype)
    context = torch.zeros(b, 512, 4096, device=device, dtype=dtype)
    clip_feature = torch.zeros(b, 257, 1280, device=device, dtype=dtype)
    action = torch.randn(b, 24, 32, device=device, dtype=dtype)
    state = torch.randn(b, 1, 64, device=device, dtype=dtype)
    embodiment_id = torch.zeros(b, device=device, dtype=torch.long)

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(args.steps, device=device, shift=5.0)
    timesteps = scheduler.timesteps

    def _dit_step_mask(n: int) -> list[bool]:
        if n == 5:
            return [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False]
        if n == 6:
            return [True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True]
        if n == 7:
            return [True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True]
        if n == 8:
            return [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True]
        return [True] * args.steps

    step_mask = _dit_step_mask(args.dit_compute_steps or args.steps)
    if len(step_mask) != args.steps:
        step_mask = [True] * args.steps

    def run_once() -> tuple[float, list[float], int]:
        kv_cache = _make_kv_cache(model, b, dtype, device)
        sample = x.clone()
        sample_action = action.clone()
        per_step = []
        dit_compute_steps = 0
        flow_pred = None
        action_pred = None
        for i, t in enumerate(timesteps):
            timestep = torch.full((b, latent_f), int(t.item()), device=device, dtype=torch.int64)
            timestep_action = torch.full((b, 24), int(t.item()), device=device, dtype=torch.int64)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if step_mask[i] or flow_pred is None or action_pred is None:
                dit_compute_steps += 1
                with torch.no_grad():
                    flow_pred, action_pred, _ = model._forward_inference(
                        x=sample,
                        timestep=timestep,
                        context=context,
                        seq_len=seq_len,
                        kv_cache=kv_cache,
                        crossattn_cache=None,
                        current_start_frame=1,
                        y=None,
                        clip_feature=clip_feature,
                        action=sample_action,
                        timestep_action=timestep_action,
                        state=state,
                        embodiment_id=embodiment_id,
                    )
            torch.cuda.synchronize()
            per_step.append(time.perf_counter() - t0)
            # 只为保持 loop 形态；核心 latency 是 DiT forward，上采样更新不改变结论。
            sample = sample - 0.0 * flow_pred
            sample_action = sample_action - 0.0 * action_pred
        return sum(per_step), per_step, dit_compute_steps

    for _ in range(args.warmup):
        run_once()

    repeats = []
    max_alloc = 0
    for _ in range(args.repeat):
        torch.cuda.reset_peak_memory_stats(device)
        total, per_step, dit_compute_steps = run_once()
        max_alloc = max(max_alloc, torch.cuda.max_memory_allocated(device))
        repeats.append({"total_sec": total, "per_step_sec": per_step, "dit_compute_steps": dit_compute_steps})

    out = {
        "scope": "DreamZero Wan2.2-TI2V-5B CausalWanModel synthetic DiT/action-register latency",
        "device": torch.cuda.get_device_name(torch.device(device)),
        "dtype": args.dtype,
        "steps": args.steps,
        "official_default_steps_in_code": 16,
        "dit_compute_steps_requested": args.dit_compute_steps,
        "dit_step_mask": step_mask,
        "num_frame_per_block": 2,
        "num_action_per_block": 24,
        "action_horizon": 24,
        "latent_shape": [b, 48, latent_f, latent_h, latent_w],
        "seq_len": seq_len,
        "load_weights": args.load_weights,
        "load_report": load_report,
        "max_memory_allocated_gib": max_alloc / (1024**3),
        "repeats": repeats,
        "mean_total_sec": sum(r["total_sec"] for r in repeats) / len(repeats),
        "mean_step_sec": sum(sum(r["per_step_sec"]) / len(r["per_step_sec"]) for r in repeats) / len(repeats),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
