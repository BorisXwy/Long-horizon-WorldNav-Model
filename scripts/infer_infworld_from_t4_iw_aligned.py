#!/usr/bin/env python3
"""Run native InfiniteWorld on the same T4 episode/window used by NAV V1 eval.

流程：
1. 从 T4 micro latent episode 取 GT history micro chunks；
2. decode/stitch 成 81-frame RGB history；
3. 用 Wan VAE 重新 encode 成 InfiniteWorld 标准 81-frame latent condition；
4. 原生 InfiniteWorld HPMC 生成一个 81-frame future chunk；
5. 保存 infworld_gen / gt_future / gt_history_tail 与 metrics。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

NAV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = NAV_ROOT.parent
INFINITE_ROOT = REPO_ROOT / "Infinite-World"
sys.path.insert(0, str(NAV_ROOT / "src"))
sys.path.insert(0, str(INFINITE_ROOT))

import infworld.context_parallel.context_parallel_util as cp_util
from infworld.models.scheduler import timestep_transform
from infworld.utils.data_utils import save_silent_video
from infworld.utils.prepare_dataloader import get_obj_from_str


MICRO_FRAMES = 13
MICRO_STRIDE = 12


def micro_steps_for_iw_chunks(iw_chunks: int) -> int:
    target_frames = 1 + 80 * iw_chunks
    if target_frames <= MICRO_FRAMES:
        return 1
    return math.ceil((target_frames - MICRO_FRAMES) / MICRO_STRIDE) + 1


def target_frames_for_iw_chunks(iw_chunks: int) -> int:
    return 1 + 80 * iw_chunks


def pad_action_81(values: torch.Tensor, start: int, length: int = 81) -> torch.Tensor:
    out = torch.zeros(81, dtype=torch.long)
    if values.numel() == 0:
        return out
    segment = values[start : min(start + length, values.numel())].long()
    if segment.numel():
        out[: segment.numel()] = segment
    return out


def latent_num_micro(path: Path) -> int:
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        return int(json.loads(sidecar.read_text())["cached_micro_chunks"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("num_micro_chunks", payload["micro_latents"].shape[1]))


def choose_example(
    data_root: Path,
    *,
    datasets: set[str],
    history_iw_chunks: int,
    future_iw_chunks: int,
    seed: int,
    explicit_episode: Path | None,
    explicit_start_micro: int | None,
) -> tuple[Path, int, int, int]:
    history_micro = micro_steps_for_iw_chunks(history_iw_chunks)
    future_micro = micro_steps_for_iw_chunks(future_iw_chunks)
    if explicit_episode is not None:
        num_micro = latent_num_micro(explicit_episode)
        start = explicit_start_micro if explicit_start_micro is not None else 0
        if num_micro < start + history_micro + future_micro:
            raise ValueError(
                f"explicit episode too short: num_micro={num_micro}, "
                f"need start+{history_micro}+{future_micro}"
            )
        return explicit_episode, num_micro, history_micro, future_micro

    rng = random.Random(seed)
    files = sorted(data_root.rglob("*.pt"))
    eligible: list[tuple[Path, int]] = []
    for path in files:
        if datasets and path.parent.name not in datasets:
            continue
        num_micro = latent_num_micro(path)
        if num_micro >= history_micro + future_micro:
            eligible.append((path, num_micro))
    if not eligible:
        raise RuntimeError("no eligible T4 episode found")
    rng.shuffle(eligible)
    path, num_micro = eligible[0]
    return path, num_micro, history_micro, future_micro


def decode_micro_chunks(
    vae,
    latents: list[torch.Tensor],
    *,
    device: torch.device,
    trim_frames: int | None = None,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    with torch.inference_mode():
        for index, latent in enumerate(latents):
            px = vae.decode(latent.to(device=device, dtype=torch.bfloat16)).cpu()
            if index > 0:
                px = px[:, :, 1:]
            pieces.append(px)
    video = torch.cat(pieces, dim=2)
    if trim_frames is not None:
        video = video[:, :, :trim_frames]
    return video


def video_metrics(px: torch.Tensor) -> dict[str, Any]:
    x = px.float().cpu()
    frames = x[0].permute(1, 2, 3, 0).numpy()
    frame_std = frames.std(axis=(1, 2, 3))
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3)) if len(frames) > 1 else np.array([])
    return {
        "num_frames": int(frames.shape[0]),
        "mean": float(frames.mean()),
        "std": float(frame_std.mean()),
        "min_frame_std": float(frame_std.min()),
        "temporal_absdiff_mean": float(diffs.mean()) if diffs.size else 0.0,
        "black_like": bool(frame_std.mean() < 0.01 or np.abs(frames + 1.0).mean() < 0.02),
        "static_like": bool((diffs.mean() if diffs.size else 0.0) < 0.002),
    }


def load_text_pair(
    sample_path: Path,
    text_cache_root: Path,
    empty_text_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    empty = torch.load(empty_text_path, map_location="cpu", weights_only=False)
    uncond_y = empty["y"].to(device=device, dtype=torch.bfloat16)
    uncond_mask = empty["y_mask"].to(device=device)
    text_path = text_cache_root / sample_path.parent.name / f"{sample_path.stem}.pt"
    if text_path.is_file():
        payload = torch.load(text_path, map_location="cpu", weights_only=False)
        cond_y = payload["y"].to(device=device, dtype=torch.bfloat16)
        cond_mask = payload["y_mask"].to(device=device)
        meta = {
            "mode": "cached_sidecar",
            "text_cache": str(text_path),
            "text": payload.get("text", ""),
            "text_source": payload.get("text_source", {}),
        }
    else:
        cond_y = uncond_y
        cond_mask = uncond_mask
        meta = {"mode": "empty", "text_cache": str(text_path), "text": ""}
    return torch.cat([cond_y, uncond_y], dim=0), torch.cat([cond_mask, uncond_mask], dim=0), meta


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
    state.pop("pos_embed_temporal", None)
    state.pop("pos_embed", None)
    result = model.load_state_dict(state, strict=False)
    print(
        json.dumps(
            {
                "event": "loaded_infworld",
                "checkpoint": str(checkpoint),
                "missing": len(result.missing_keys),
                "unexpected": len(result.unexpected_keys),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return model.to(device=device, dtype=torch.bfloat16).eval()


def make_timesteps(steps: int, shift: float, device: torch.device) -> list[torch.Tensor]:
    raw = list(np.linspace(1000, 1, steps, dtype=np.float32))
    return [
        timestep_transform(
            torch.tensor([value], device=device),
            shift=float(shift),
            num_timesteps=1000,
        )
        for value in raw
    ]


def rflow_sample_infworld(
    model,
    z_shape: tuple[int, ...],
    timesteps: list[torch.Tensor],
    model_kwargs: dict[str, torch.Tensor],
    *,
    guidance_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    z = torch.randn(*z_shape, device=device, dtype=torch.bfloat16)
    forward_times: list[float] = []
    for index, timestep in enumerate(timesteps):
        z_in = torch.cat([z, z], dim=0)
        t_in = torch.cat([timestep, timestep], dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize(device.index or 0)
        start = time.perf_counter()
        with torch.inference_mode():
            pred = model(x=z_in, t=t_in, **model_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device.index or 0)
        forward_times.append(time.perf_counter() - start)
        pred = pred[:, :, -z_in.shape[2] :]
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        v_pred = -(pred_uncond + guidance_scale * (pred_cond - pred_uncond))
        dt = timesteps[index] - timesteps[index + 1] if index < len(timesteps) - 1 else timesteps[index]
        dt = dt / 1000.0
        z = z + v_pred * dt[:, None, None, None, None]
    return z, forward_times


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("/sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt"))
    p.add_argument("--data-root", type=Path, default=Path("/sharedata/NAV/derived/v1/t4_micro_latents_spatial20"))
    p.add_argument("--episode", type=Path, default=None)
    p.add_argument("--start-micro", type=int, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="spatialvid")
    p.add_argument("--history-iw-chunks", type=int, default=1)
    p.add_argument("--future-iw-chunks", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--text-cache-root", type=Path, default=Path("/sharedata/NAV/derived/v1/text_embeddings/t4_micro"))
    p.add_argument("--empty-text", type=Path, default=Path("/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cp_util.dp_rank = cp_util.cp_rank = 0
    cp_util.dp_size = cp_util.cp_size = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    config = OmegaConf.load(INFINITE_ROOT / "configs/infworld_config.yaml")
    vae = get_obj_from_str(config.vae_target)(**config.vae_cfg).to(device)
    model = build_infworld(config, args.checkpoint, device)
    timesteps = make_timesteps(args.steps, float(config.val_scheduler_cfg.shift), device)

    datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    episode_path, num_micro, history_micro, future_micro = choose_example(
        args.data_root,
        datasets=datasets,
        history_iw_chunks=args.history_iw_chunks,
        future_iw_chunks=args.future_iw_chunks,
        seed=args.seed,
        explicit_episode=args.episode,
        explicit_start_micro=args.start_micro,
    )
    start_micro = args.start_micro
    if start_micro is None:
        max_start = num_micro - history_micro - future_micro
        rng = random.Random(args.seed)
        start_micro = rng.randint(0, max_start) if max_start > 0 else 0

    sample_dir = args.output_dir / (
        f"infworld_{episode_path.parent.name}_histIW{args.history_iw_chunks}_"
        f"futureIW{args.future_iw_chunks}_{episode_path.stem}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "sample_start",
                "episode": str(episode_path),
                "sample_dir": str(sample_dir),
                "num_micro": num_micro,
                "history_micro_steps": history_micro,
                "future_micro_steps": future_micro,
                "start_micro": start_micro,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    payload = torch.load(episode_path, map_location="cpu", weights_only=False)
    chunks = payload["micro_latents"]
    move_all = payload.get("move", torch.zeros(0, dtype=torch.long)).long()
    view_all = payload.get("view", torch.zeros(0, dtype=torch.long)).long()
    history_latents = [
        chunks[:, start_micro + i].to(device=device, dtype=torch.bfloat16)
        for i in range(history_micro)
    ]
    future_latents = [
        chunks[:, start_micro + history_micro + i].to(device=device, dtype=torch.bfloat16)
        for i in range(future_micro)
    ]
    target_frames = target_frames_for_iw_chunks(args.future_iw_chunks)
    history_frames = target_frames_for_iw_chunks(args.history_iw_chunks)

    gt_history_px = decode_micro_chunks(vae, history_latents, device=device, trim_frames=history_frames)
    gt_future_px = decode_micro_chunks(vae, future_latents, device=device, trim_frames=target_frames)
    with torch.inference_mode():
        history_cond = vae.encode(gt_history_px.to(device=device, dtype=torch.bfloat16)).to(torch.bfloat16)
    print(
        json.dumps(
            {
                "event": "encoded_history",
                "gt_history_px": list(gt_history_px.shape),
                "history_cond": list(history_cond.shape),
                "gt_future_px": list(gt_future_px.shape),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    y, y_mask, text_meta = load_text_pair(episode_path, args.text_cache_root, args.empty_text, device)
    future_frame_start = (start_micro + history_micro) * MICRO_STRIDE
    move = pad_action_81(move_all, future_frame_start, 81).to(device)[None]
    view = pad_action_81(view_all, future_frame_start, 81).to(device)[None]
    z_shape = tuple(history_cond.shape)
    model_kwargs = {
        "y": y,
        "y_mask": y_mask,
        "image_cond": history_cond.repeat(2, 1, 1, 1, 1),
        "move": move.repeat(2, 1),
        "view": view.repeat(2, 1),
    }
    start = time.perf_counter()
    sample, forward_times = rflow_sample_infworld(
        model,
        z_shape,
        timesteps,
        model_kwargs,
        guidance_scale=args.cfg_scale,
        device=device,
    )
    sample_s = time.perf_counter() - start
    with torch.inference_mode():
        gen_px = vae.decode(sample).cpu()

    save_silent_video(gen_px.to(device), str(sample_dir / "infworld_gen"), fps=30, quality=10)
    save_silent_video(gt_future_px.to(device), str(sample_dir / "gt_future"), fps=30, quality=10)
    save_silent_video(gt_history_px.to(device), str(sample_dir / "gt_history"), fps=30, quality=10)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "episode": str(episode_path),
        "sample_dir": str(sample_dir),
        "num_micro": num_micro,
        "history_iw_chunks": args.history_iw_chunks,
        "future_iw_chunks": args.future_iw_chunks,
        "history_micro_steps": history_micro,
        "future_micro_steps": future_micro,
        "start_micro": start_micro,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "text": text_meta,
        "sample_total_s": sample_s,
        "mean_forward_s": float(np.mean(forward_times)),
        "num_denoise_steps": len(forward_times),
        "infworld_gen": video_metrics(gen_px),
        "gt_future": video_metrics(gt_future_px),
        "latent_l1": float((sample.float().cpu() - vae.encode(gt_future_px.to(device=device, dtype=torch.bfloat16)).float().cpu()).abs().mean()),
    }
    (sample_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    summary = {"num_examples": 1, "examples": [metrics]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
