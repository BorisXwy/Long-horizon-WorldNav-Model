#!/usr/bin/env python3
"""V1 T4 autoregressive IW-aligned generation.

用 GT history 初始化/更新 Register，然后连续生成多个 T4 micro chunks；
每个已生成 latent 会作为下一步 history 回灌 Register，直到 future 视频长度
对齐指定数量的 InfiniteWorld 81-frame chunks。

输出:
  - nav_autoreg.mp4: NAV 自回归生成 future，按 micro stride stitch 并 trim；
  - gt_future.mp4: 同一时间范围的 GT future；
  - gt_history_tail.mp4: 最后一段 GT history，便于肉眼看条件；
  - metrics.json / summary.json。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
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
from nav.infinite_adapter import InfiniteRegisterAdapter


MICRO_FRAMES = 13
MICRO_STRIDE = 12
IW_CHUNK_FRAMES = 81


@dataclass
class Example:
    path: Path
    dataset: str
    num_micro: int
    history_iw_chunks: int
    history_micro_steps: int
    future_iw_chunks: int
    future_micro_steps: int
    start_micro: int


class CFGRegisterWrapper(torch.nn.Module):
    """Keep the call signature used by the custom RFlow sampler."""

    def __init__(self, model: InfiniteRegisterAdapter) -> None:
        super().__init__()
        self.model = model

    def forward(self, x, t, **kwargs):
        return self.model(x=x, t=t, **kwargs)


def micro_steps_for_iw_chunks(iw_chunks: int) -> int:
    target_frames = 1 + 80 * iw_chunks
    if target_frames <= MICRO_FRAMES:
        return 1
    return math.ceil((target_frames - MICRO_FRAMES) / MICRO_STRIDE) + 1


def target_frames_for_iw_chunks(iw_chunks: int) -> int:
    return 1 + 80 * iw_chunks


def parse_csv_ints(text: str) -> list[int]:
    values = [int(x) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("需要至少一个整数")
    return values


def parse_csv_names(text: str) -> set[str]:
    return {x.strip() for x in text.split(",") if x.strip()}


def latent_num_micro(path: Path) -> int:
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        return int(json.loads(sidecar.read_text())["cached_micro_chunks"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("num_micro_chunks", payload["micro_latents"].shape[1]))


def choose_examples(
    data_root: Path,
    history_iw_chunks: list[int],
    future_iw_chunks: int,
    samples_per_history: int,
    datasets: set[str],
    seed: int,
) -> list[Example]:
    rng = random.Random(seed)
    files = sorted(data_root.rglob("*.pt"))
    examples: list[Example] = []
    for history_iw in history_iw_chunks:
        history_micro = micro_steps_for_iw_chunks(history_iw)
        future_micro = micro_steps_for_iw_chunks(future_iw_chunks)
        eligible: list[tuple[Path, int, str]] = []
        for path in files:
            dataset = path.parent.name
            if datasets and dataset not in datasets:
                continue
            num_micro = latent_num_micro(path)
            if num_micro >= history_micro + future_micro:
                eligible.append((path, num_micro, dataset))
        if not eligible:
            print(
                json.dumps(
                    {
                        "event": "no_eligible_example",
                        "history_iw_chunks": history_iw,
                        "history_micro_steps": history_micro,
                        "future_iw_chunks": future_iw_chunks,
                        "future_micro_steps": future_micro,
                        "datasets": sorted(datasets),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        rng.shuffle(eligible)
        for path, num_micro, dataset in eligible[:samples_per_history]:
            max_start = num_micro - history_micro - future_micro
            start_micro = rng.randint(0, max_start) if max_start > 0 else 0
            examples.append(
                Example(
                    path=path,
                    dataset=dataset,
                    num_micro=num_micro,
                    history_iw_chunks=history_iw,
                    history_micro_steps=history_micro,
                    future_iw_chunks=future_iw_chunks,
                    future_micro_steps=future_micro,
                    start_micro=start_micro,
                )
            )
    return examples


def pad_action_81(values: torch.Tensor, start: int, length: int) -> torch.Tensor:
    out = torch.zeros(81, dtype=torch.long)
    if values.numel() == 0:
        return out
    segment = values[start : min(start + length, values.numel())].long()
    if segment.numel():
        out[-segment.numel() :] = segment
    return out


def build_model(checkpoint: Path, device: torch.device) -> InfiniteRegisterAdapter:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    variant = state.get("variant", "latent_prefix")
    if variant != "latent_prefix":
        raise ValueError(f"当前 IW-aligned autoreg 脚本只支持 A/latent_prefix，checkpoint variant={variant}")
    model = InfiniteRegisterAdapter(
        variant=variant,
        model_cfg={
            "model_type": "t2v",
            "dim": 1536,
            "in_channels": 16,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "num_heads": 12,
            "num_layers": 30,
            "out_channels": 16,
            "caption_channels": 4096,
            "model_max_length": 512,
        },
        register_cfg={
            "channels": 16,
            "register_frames": 4,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_update_layers": 2,
            "chunk_time_tokens": 4,
        },
    )
    model.remove_hmpc()
    current_backbone = model.backbone.state_dict()
    raw_backbone = state["backbone"]
    filtered_backbone = {
        key: value
        for key, value in raw_backbone.items()
        if key in current_backbone and tuple(value.shape) == tuple(current_backbone[key].shape)
    }
    model.backbone.load_state_dict(filtered_backbone, strict=False)
    model.register_memory.load_state_dict(state["register_memory"], strict=True)
    if "action_interface" in state:
        model.action_interface.load_state_dict(state["action_interface"], strict=False)
    if "shared_action" in state:
        model.load_shared_action_state_dict(state["shared_action"])
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
    meta: dict[str, Any] = {
        "mode": "empty",
        "text_cache": str(text_path),
        "text": "",
    }
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
    y = torch.cat([cond_y, uncond_y], dim=0)
    y_mask = torch.cat([cond_mask, uncond_mask], dim=0)
    return y, y_mask, meta


def rflow_sample_one_micro(
    wrapper: CFGRegisterWrapper,
    z_shape: tuple[int, ...],
    timesteps: list[torch.Tensor],
    model_kwargs: dict[str, torch.Tensor],
    *,
    guidance_scale: float,
    device: torch.device,
    progress: bool,
) -> tuple[torch.Tensor, list[float]]:
    z = torch.randn(*z_shape, device=device, dtype=torch.bfloat16)
    forward_times: list[float] = []
    iterable = enumerate(timesteps)
    for index, timestep in iterable:
        z_in = torch.cat([z, z], dim=0)
        t_in = torch.cat([timestep, timestep], dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize(device.index or 0)
        start = time.perf_counter()
        with torch.inference_mode():
            pred = wrapper(z_in, t_in, **model_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device.index or 0)
        forward_times.append(time.perf_counter() - start)
        pred = pred[:, :, -z_in.shape[2] :]
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        v_pred = -(pred_uncond + guidance_scale * (pred_cond - pred_uncond))
        dt = timesteps[index] - timesteps[index + 1] if index < len(timesteps) - 1 else timesteps[index]
        dt = dt / 1000.0
        z = z + v_pred * dt[:, None, None, None, None]
        if progress:
            print(
                json.dumps(
                    {
                        "event": "denoise_step",
                        "step": index + 1,
                        "num_steps": len(timesteps),
                        "forward_s": forward_times[-1],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return z, forward_times


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
            # Consecutive micro chunks overlap by one RGB frame because stride=12
            # and each decoded micro chunk has 13 RGB frames.
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


def save_video(px: torch.Tensor, path_without_ext: Path, *, device: torch.device) -> None:
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    save_silent_video(px.to(device), str(path_without_ext), fps=30, quality=10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/sharedata/NAV/derived/v1/t4_micro_latents_spatial20"),
    )
    p.add_argument(
        "--text-cache-root",
        type=Path,
        default=Path("/sharedata/NAV/derived/v1/text_embeddings/t4_micro"),
    )
    p.add_argument(
        "--empty-text",
        type=Path,
        default=Path("/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="spatialvid,dl3dv,re10k")
    p.add_argument("--history-iw-chunks", default="1")
    p.add_argument("--future-iw-chunks", type=int, default=1)
    p.add_argument("--samples-per-history", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--save-history-tail-micro", type=int, default=7)
    p.add_argument("--progress", action="store_true")
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
    timesteps = make_timesteps(args.steps, float(config.val_scheduler_cfg.shift), device)
    model = build_model(args.checkpoint, device)
    wrapper = CFGRegisterWrapper(model)

    examples = choose_examples(
        args.data_root,
        parse_csv_ints(args.history_iw_chunks),
        args.future_iw_chunks,
        args.samples_per_history,
        parse_csv_names(args.datasets),
        args.seed,
    )
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "device": args.device,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "datasets": sorted(parse_csv_names(args.datasets)),
        "history_iw_chunks": parse_csv_ints(args.history_iw_chunks),
        "future_iw_chunks": args.future_iw_chunks,
        "future_micro_steps": micro_steps_for_iw_chunks(args.future_iw_chunks),
        "future_target_frames": target_frames_for_iw_chunks(args.future_iw_chunks),
        "examples": [],
    }

    for index, ex in enumerate(examples):
        sample_dir = args.output_dir / (
            f"{index:03d}_{ex.dataset}_histIW{ex.history_iw_chunks}_"
            f"futureIW{ex.future_iw_chunks}_{ex.path.stem}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "event": "sample_start",
                    "index": index,
                    "sample_dir": str(sample_dir),
                    **ex.__dict__,
                    "path": str(ex.path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        payload = torch.load(ex.path, map_location="cpu", weights_only=False)
        chunks = payload["micro_latents"]
        move_all = payload.get("move", torch.zeros(0, dtype=torch.long)).long()
        view_all = payload.get("view", torch.zeros(0, dtype=torch.long)).long()
        y, y_mask, text_meta = load_text_pair(
            ex.path, args.text_cache_root, args.empty_text, device
        )

        history_latents: list[torch.Tensor] = []
        history_moves: list[torch.Tensor] = []
        history_views: list[torch.Tensor] = []
        for offset in range(ex.history_micro_steps):
            micro_index = ex.start_micro + offset
            history_latents.append(
                chunks[:, micro_index].to(device=device, dtype=torch.bfloat16)
            )
            frame_start = micro_index * MICRO_STRIDE
            history_moves.append(pad_action_81(move_all, frame_start, MICRO_FRAMES).to(device)[None])
            history_views.append(pad_action_81(view_all, frame_start, MICRO_FRAMES).to(device)[None])

        conditioned = model.action_interface.condition_history_chunk(
            history_latents[0], history_moves[0], history_views[0]
        )
        registers = model.register_memory.extract(conditioned)
        for hist, h_move, h_view in zip(
            history_latents[1:], history_moves[1:], history_views[1:]
        ):
            conditioned = model.action_interface.condition_history_chunk(hist, h_move, h_view)
            registers = model.register_memory.update(registers, conditioned)
        local_latent = history_latents[-1][:, :, -1:]

        generated_latents: list[torch.Tensor] = []
        gt_future_latents: list[torch.Tensor] = []
        all_forward_times: list[float] = []
        chunk_times: list[dict[str, Any]] = []
        for future_offset in range(ex.future_micro_steps):
            micro_index = ex.start_micro + ex.history_micro_steps + future_offset
            gt_future = chunks[:, micro_index].to(device=device, dtype=torch.bfloat16)
            gt_future_latents.append(gt_future)
            frame_start = micro_index * MICRO_STRIDE
            move = pad_action_81(move_all, frame_start, MICRO_FRAMES).to(device)[None]
            view = pad_action_81(view_all, frame_start, MICRO_FRAMES).to(device)[None]
            model_kwargs = {
                "y": y,
                "y_mask": y_mask,
                "registers": registers.repeat(2, 1, 1, 1, 1),
                "local_latent": local_latent.repeat(2, 1, 1, 1, 1),
                "move": move.repeat(2, 1),
                "view": view.repeat(2, 1),
            }
            start_time = time.perf_counter()
            sample, forward_times = rflow_sample_one_micro(
                wrapper,
                tuple(gt_future.shape),
                timesteps,
                model_kwargs,
                guidance_scale=args.cfg_scale,
                device=device,
                progress=args.progress,
            )
            chunk_s = time.perf_counter() - start_time
            generated_latents.append(sample.detach())
            all_forward_times.extend(forward_times)
            chunk_times.append(
                {
                    "future_offset": future_offset,
                    "micro_index": micro_index,
                    "sample_total_s": chunk_s,
                    "mean_forward_s": float(np.mean(forward_times)),
                    "num_denoise_steps": len(forward_times),
                }
            )
            # Autoregressive path: update Register/local memory with generated latent.
            conditioned_generated = model.action_interface.condition_history_chunk(
                sample, move, view
            )
            registers = model.register_memory.update(registers, conditioned_generated)
            local_latent = sample[:, :, -1:]
            print(
                json.dumps(
                    {
                        "event": "generated_micro",
                        "index": index,
                        **chunk_times[-1],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        target_frames = target_frames_for_iw_chunks(ex.future_iw_chunks)
        history_tail = history_latents[-args.save_history_tail_micro :]
        with torch.inference_mode():
            nav_px = decode_micro_chunks(
                vae, generated_latents, device=device, trim_frames=target_frames
            )
            gt_future_px = decode_micro_chunks(
                vae, gt_future_latents, device=device, trim_frames=target_frames
            )
            history_tail_px = decode_micro_chunks(vae, history_tail, device=device)

        save_video(nav_px, sample_dir / "nav_autoreg", device=device)
        save_video(gt_future_px, sample_dir / "gt_future", device=device)
        save_video(history_tail_px, sample_dir / "gt_history_tail", device=device)
        metrics = {
            **ex.__dict__,
            "path": str(ex.path),
            "sample_dir": str(sample_dir),
            "text": text_meta,
            "target_frames": target_frames,
            "chunk_times": chunk_times,
            "mean_forward_s": float(np.mean(all_forward_times)) if all_forward_times else 0.0,
            "mean_micro_sample_s": float(np.mean([x["sample_total_s"] for x in chunk_times])) if chunk_times else 0.0,
            "nav_autoreg": video_metrics(nav_px),
            "gt_future": video_metrics(gt_future_px),
            "latent_l1_by_micro": [
                float((gen.float().cpu() - gt.float().cpu()).abs().mean())
                for gen, gt in zip(generated_latents, gt_future_latents)
            ],
        }
        (sample_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
        )
        summary["examples"].append(metrics)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary["num_examples"] = len(summary["examples"])
    summary["num_black_like"] = sum(
        1 for item in summary["examples"] if item["nav_autoreg"]["black_like"]
    )
    summary["num_static_like"] = sum(
        1 for item in summary["examples"] if item["nav_autoreg"]["static_like"]
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
