#!/usr/bin/env python3
"""Measure one-chunk inference latency for NAV A/B and Infinite-World.

The probe uses cached Wan VAE latents, so the measured stages are:

1. history preparation: Register extract/update for NAV, HPMC compression for
   Infinite-World;
2. one DiT forward on a CFG batch of 2;
3. full chunk sampling with N DiT forwards;
4. Wan VAE decode of the generated latent chunk.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

NAV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = NAV_ROOT.parent
INFINITE_ROOT = REPO_ROOT / "Infinite-World"
sys.path.insert(0, str(NAV_ROOT / "src"))
sys.path.insert(0, str(INFINITE_ROOT))

import infworld.context_parallel.context_parallel_util as cp_util
from infworld.utils.prepare_dataloader import get_obj_from_str
from nav.infinite_adapter import InfiniteRegisterAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        type=Path,
        default=Path(
            "/sharedata/NAV/derived/latents/full_episodes_v1/dl3dv/"
            "dl3dv__7f8a82be9d0fa029221d6873f60e1ea7f3cb4d5f170dc7b07196ee7b3099f663.pt"
        ),
    )
    parser.add_argument(
        "--a-checkpoint",
        type=Path,
        default=NAV_ROOT
        / "log/stage-one-v1-from-scratch-a-ebs16-1000/full-step-000500.pt",
    )
    parser.add_argument(
        "--b-checkpoint",
        type=Path,
        default=NAV_ROOT
        / "log/stage-one-v1-from-scratch-b-ebs16-1000/full-final.pt",
    )
    parser.add_argument(
        "--inf-checkpoint",
        type=Path,
        default=Path("/sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt"),
    )
    parser.add_argument("--history-chunk", type=int, default=0)
    parser.add_argument("--target-chunk", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=NAV_ROOT / "result/latency/inference_latency_latest_ab_infworld.json",
    )
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device.index or 0)


def timed(device: torch.device, fn):
    sync(device)
    start = time.perf_counter()
    out = fn()
    sync(device)
    return time.perf_counter() - start, out


def summarize(values: list[float]) -> dict:
    return {
        "values_s": values,
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def build_nav(variant: str, checkpoint: Path, device: torch.device):
    model = InfiniteRegisterAdapter(
        variant=variant,
        model_cfg={
            "model_type": "t2v",
            "dim": 1536,
            "in_channels": 20,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "num_heads": 12,
            "num_layers": 30,
            "out_channels": 16,
            "caption_channels": 4096,
            "model_max_length": 512,
        },
        register_cfg=(
            {
                "channels": 16,
                "register_frames": 4,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_update_layers": 2,
                "chunk_time_tokens": 8,
            }
            if variant == "latent_prefix"
            else {
                "latent_channels": 16,
                "register_dim": 256,
                "num_registers": 16,
                "num_update_layers": 2,
                "num_heads": 8,
                "caption_channels": 4096,
            }
        ),
    )
    model.remove_hmpc()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(state["backbone"], strict=True)
    model.register_memory.load_state_dict(state["register_memory"], strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval(), state.get("step")


def build_infworld(config, checkpoint: Path, device: torch.device):
    model = get_obj_from_str(config.model_target)(
        out_channels=16,
        caption_channels=4096,
        model_max_length=512,
        enable_context_parallel=False,
        **config.model_cfg,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def load_conditions(device: torch.device):
    text = torch.load(
        "/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt",
        map_location="cpu",
        weights_only=False,
    )
    y = text["y"].to(device=device, dtype=torch.bfloat16).repeat(2, 1, 1, 1)
    y_mask = text["y_mask"].to(device=device).repeat(2, 1)
    return y, y_mask


def rflow_loop(model, z, timesteps, model_kwargs, guidance_scale=5.0):
    current = z
    for index, timestep in enumerate(timesteps):
        z_in = torch.cat([current, current], dim=0)
        t_in = torch.cat([timestep, timestep], dim=0)
        pred = model(x=z_in, t=t_in, **model_kwargs)
        pred = pred[:, :, -z_in.shape[2] :]
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        velocity = -(pred_uncond + guidance_scale * (pred_cond - pred_uncond))
        dt = timesteps[index] - timesteps[index + 1] if index < len(timesteps) - 1 else timesteps[index]
        dt = dt / 1000.0
        current = current + velocity * dt[:, None, None, None, None]
    return current


def make_timesteps(config, steps: int, device: torch.device):
    import numpy as np
    from infworld.models.scheduler import timestep_transform

    raw = list(np.linspace(1000, 1, steps, dtype=np.float32))
    return [
        timestep_transform(
            torch.tensor([value], device=device),
            shift=float(config.val_scheduler_cfg.shift),
            num_timesteps=1000,
        )
        for value in raw
    ]


def main() -> None:
    args = parse_args()
    cp_util.dp_rank = cp_util.cp_rank = 0
    cp_util.dp_size = cp_util.cp_size = 1
    torch.manual_seed(1234)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    config = OmegaConf.load(INFINITE_ROOT / "configs/infworld_config.yaml")
    vae = get_obj_from_str(config.vae_target)(**config.vae_cfg).to(device)
    y, y_mask = load_conditions(device)
    episode = torch.load(args.episode, map_location="cpu", weights_only=False)
    history = episode["chunks"][0, args.history_chunk].to(
        device=device, dtype=torch.bfloat16
    )[None]
    target = episode["chunks"][0, args.target_chunk].to(
        device=device, dtype=torch.bfloat16
    )[None]
    action_start = args.target_chunk * 81
    move = episode["move"][action_start : action_start + 81].long().to(device)
    view = episode["view"][action_start : action_start + 81].long().to(device)
    move2 = move[None].repeat(2, 1)
    view2 = view[None].repeat(2, 1)
    z_template = torch.randn_like(target)
    timesteps = make_timesteps(config, args.steps, device)

    results: dict[str, dict] = {
        "metadata": {
            "episode": str(args.episode),
            "history_chunk": args.history_chunk,
            "target_chunk": args.target_chunk,
            "sampling_steps": args.steps,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "device": args.device,
            "latent_shape": list(target.shape),
            "note": (
                "single_forward is one CFG DiT call. sampling_total is "
                "steps CFG DiT calls with cached empty text embedding; "
                "VAE encode and text encode are excluded."
            ),
        }
    }

    models = [
        ("nav_a", "latent_prefix", args.a_checkpoint),
        ("nav_b", "dit_condition", args.b_checkpoint),
        ("infworld", "infworld", args.inf_checkpoint),
    ]

    for name, variant, checkpoint in models:
        print(f"[latency] loading {name}: {checkpoint}", flush=True)
        if variant == "infworld":
            model = build_infworld(config, checkpoint, device)
            model_step = None
        else:
            model, model_step = build_nav(variant, checkpoint, device)

        def prepare():
            if variant == "infworld":
                local = history[:, :, -1:]
                compressed = model.latent_encoder(history)
                return {
                    "image_cond": compressed,
                    "local_latent": local,
                    "registers": None,
                }
            registers = model.register_memory.extract(history)
            return {
                "image_cond": registers,
                "local_latent": history[:, :, -1:],
                "registers": registers,
            }

        prep_time, prepared = timed(device, prepare)
        print(f"[latency] {name} first prep {prep_time:.4f}s", flush=True)

        if variant == "infworld":
            forward_kwargs = {
                "y": y,
                "y_mask": y_mask,
                "image_cond": prepared["image_cond"].repeat(2, 1, 1, 1, 1),
                "local_memory": prepared["local_latent"].repeat(2, 1, 1, 1, 1),
                "memory_is_precomputed": True,
                "move": move2,
                "view": view2,
            }
            native_kwargs = {
                "y": y,
                "y_mask": y_mask,
                "image_cond": history.repeat(2, 1, 1, 1, 1),
                "move": move2,
                "view": view2,
            }
        else:
            registers = prepared["registers"].repeat(
                2, *([1] * (prepared["registers"].ndim - 1))
            )
            forward_kwargs = {
                "y": y,
                "y_mask": y_mask,
                "registers": registers,
                "local_latent": prepared["local_latent"].repeat(2, 1, 1, 1, 1),
                "move": move2,
                "view": view2,
            }
            native_kwargs = None

        def single_forward():
            z = torch.randn_like(z_template)
            z_in = torch.cat([z, z], dim=0)
            t_in = torch.full((2,), 500.0, device=device)
            return model(x=z_in, t=t_in, **forward_kwargs)

        def sampling():
            z = z_template.clone()
            return rflow_loop(model, z, timesteps, forward_kwargs)

        def decode(latent):
            return vae.decode(latent)

        for _ in range(args.warmup):
            with torch.inference_mode():
                single_forward()
                sample = sampling()
                decode(sample)
                if native_kwargs is not None:
                    z = torch.randn_like(z_template)
                    model(
                        x=torch.cat([z, z], dim=0),
                        t=torch.full((2,), 500.0, device=device),
                        **native_kwargs,
                    )
            sync(device)

        prep_values: list[float] = []
        forward_values: list[float] = []
        sampling_values: list[float] = []
        decode_values: list[float] = []
        native_values: list[float] = []

        with torch.inference_mode():
            for repeat in range(args.repeats):
                prep_s, prepared = timed(device, prepare)
                prep_values.append(prep_s)
                if variant == "infworld":
                    forward_kwargs["image_cond"] = prepared["image_cond"].repeat(2, 1, 1, 1, 1)
                    forward_kwargs["local_memory"] = prepared["local_latent"].repeat(2, 1, 1, 1, 1)
                else:
                    forward_kwargs["registers"] = prepared["registers"].repeat(
                        2, *([1] * (prepared["registers"].ndim - 1))
                    )
                    forward_kwargs["local_latent"] = prepared["local_latent"].repeat(2, 1, 1, 1, 1)

                forward_s, _ = timed(device, single_forward)
                forward_values.append(forward_s)
                sampling_s, sample = timed(device, sampling)
                sampling_values.append(sampling_s)
                decode_s, _ = timed(device, lambda: decode(sample))
                decode_values.append(decode_s)

                if native_kwargs is not None:
                    native_s, _ = timed(
                        device,
                        lambda: model(
                            x=torch.cat([torch.randn_like(z_template), torch.randn_like(z_template)], dim=0),
                            t=torch.full((2,), 500.0, device=device),
                            **native_kwargs,
                        ),
                    )
                    native_values.append(native_s)

                print(
                    f"[latency] {name} repeat={repeat} prep={prep_s:.4f}s "
                    f"forward={forward_s:.4f}s sampling={sampling_s:.4f}s "
                    f"decode={decode_s:.4f}s",
                    flush=True,
                )

        peak_gib = None
        entry = {
            "checkpoint": str(checkpoint),
            "checkpoint_step": model_step,
            "history_prep": summarize(prep_values),
            "single_cfg_dit_forward": summarize(forward_values),
            "sampling_total": summarize(sampling_values),
            "sampling_per_cfg_dit_forward_est": summarize(
                [value / args.steps for value in sampling_values]
            ),
            "vae_decode": summarize(decode_values),
            "end_to_end_no_text_no_video_write": summarize(
                [
                    prep_values[index] + sampling_values[index] + decode_values[index]
                    for index in range(args.repeats)
                ]
            ),
            "peak_gpu_allocated_gib_seen_so_far": peak_gib,
        }
        if native_values:
            entry["native_forward_with_hmpc_inside"] = summarize(native_values)
        results[name] = entry

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[latency] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
