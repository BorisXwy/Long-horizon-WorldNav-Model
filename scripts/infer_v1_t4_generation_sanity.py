#!/usr/bin/env python3
"""V1 Stage One T4 generation sanity check.

This script is intentionally small and direct: it samples a few cached T4
episodes, rolls the Register over an IW-equivalent history span, generates the
next T4 micro chunk, decodes both generated and GT latents, writes videos, and
records simple black-frame / temporal-change metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

NAV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = NAV_ROOT.parent
INFINITE_ROOT = REPO_ROOT / "Infinite-World"
sys.path.insert(0, str(NAV_ROOT / "src"))
sys.path.insert(0, str(INFINITE_ROOT))

import infworld.context_parallel.context_parallel_util as cp_util
from infworld.utils.data_utils import save_silent_video
from infworld.utils.prepare_dataloader import get_obj_from_str
from nav.infinite_adapter import InfiniteRegisterAdapter


def micro_steps_for_iw_chunks(
    iw_chunks: int, *, micro_frames: int = 13, micro_stride: int = 12
) -> int:
    target_frames = 1 + 80 * iw_chunks
    if target_frames <= micro_frames:
        return 1
    return math.ceil((target_frames - micro_frames) / micro_stride) + 1


def pad_action_81(values: torch.Tensor, start: int, length: int) -> torch.Tensor:
    out = torch.zeros(81, dtype=torch.long)
    if values.numel() == 0:
        return out
    segment = values[start : min(start + length, values.numel())].long()
    if segment.numel():
        out[-segment.numel() :] = segment
    return out


def parse_history(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def latent_num_micro(path: Path) -> int:
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        return int(json.loads(sidecar.read_text())["cached_micro_chunks"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("num_micro_chunks", payload["micro_latents"].shape[1]))


def choose_examples(
    data_root: Path,
    history_iw_chunks: list[int],
    samples_per_history: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    files = sorted(data_root.rglob("*.pt"))
    examples: list[dict[str, Any]] = []
    for iw in history_iw_chunks:
        history_micro = micro_steps_for_iw_chunks(iw)
        eligible = []
        for path in files:
            num_micro = latent_num_micro(path)
            if num_micro >= history_micro + 1:
                eligible.append((path, num_micro))
        rng.shuffle(eligible)
        for path, num_micro in eligible[:samples_per_history]:
            max_start = num_micro - history_micro - 1
            start_micro = rng.randint(0, max_start) if max_start > 0 else 0
            examples.append(
                {
                    "path": path,
                    "history_iw_chunks": iw,
                    "history_micro_steps": history_micro,
                    "start_micro": start_micro,
                }
            )
    return examples


class CFGRegisterWrapper(nn.Module):
    def __init__(self, model: InfiniteRegisterAdapter) -> None:
        super().__init__()
        self.model = model

    @property
    def y_embedder(self):
        return self.model.backbone.y_embedder

    def forward(self, x, t, *, image_cond=None, **kwargs):
        del image_cond
        return self.model(x=x, t=t, **kwargs)


def build_model(checkpoint: Path, device: torch.device) -> InfiniteRegisterAdapter:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    variant = state.get("variant", "latent_prefix")
    if variant != "latent_prefix":
        raise ValueError(f"T4 sanity 当前只支持 latent_prefix，checkpoint variant={variant}")
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


def build_registers(
    model: InfiniteRegisterAdapter,
    chunks: torch.Tensor,
    move_all: torch.Tensor,
    view_all: torch.Tensor,
    start_micro: int,
    history_micro: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    histories = []
    for offset in range(history_micro):
        histories.append(
            chunks[:, start_micro + offset]
            .to(device=device, dtype=torch.bfloat16)
        )
    history_moves = []
    history_views = []
    for offset in range(history_micro):
        frame_start = (start_micro + offset) * 12
        history_moves.append(pad_action_81(move_all, frame_start, 13).to(device)[None])
        history_views.append(pad_action_81(view_all, frame_start, 13).to(device)[None])
    target_frame_start = (start_micro + history_micro) * 12
    move = pad_action_81(move_all, target_frame_start, 13).to(device)[None]
    view = pad_action_81(view_all, target_frame_start, 13).to(device)[None]

    conditioned = model.action_interface.condition_history_chunk(
        histories[0], history_moves[0], history_views[0]
    )
    registers = model.register_memory.extract(conditioned)
    for history, h_move, h_view in zip(
        histories[1:], history_moves[1:], history_views[1:]
    ):
        conditioned = model.action_interface.condition_history_chunk(
            history, h_move, h_view
        )
        registers = model.register_memory.update(registers, conditioned)
    local_latent = histories[-1][:, :, -1:]
    target = chunks[:, start_micro + history_micro].to(
        device=device, dtype=torch.bfloat16
    )
    return registers, local_latent, move, view, target


def video_metrics(px: torch.Tensor) -> dict[str, Any]:
    # px shape is expected as [B,C,T,H,W], roughly in [-1,1].
    x = px.float().cpu()
    frames = x[0].permute(1, 2, 3, 0).numpy()
    frame_mean = frames.mean(axis=(1, 2, 3)).tolist()
    frame_std = frames.std(axis=(1, 2, 3)).tolist()
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3)).tolist()
    return {
        "num_frames": int(frames.shape[0]),
        "mean": float(np.mean(frame_mean)),
        "std": float(np.mean(frame_std)),
        "min_frame_std": float(np.min(frame_std)),
        "temporal_absdiff_mean": float(np.mean(diffs)) if diffs else 0.0,
        "frame_mean": frame_mean,
        "frame_std": frame_std,
        "temporal_absdiff": diffs,
        "black_like": bool(np.mean(frame_std) < 0.01 or np.mean(np.abs(frames + 1.0)) < 0.02),
        "static_like": bool((np.mean(diffs) if diffs else 0.0) < 0.002),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/sharedata/NAV/derived/v1/t4_micro_latents_spatial20"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--history-iw-chunks", default="1,4,8,16")
    p.add_argument("--samples-per-history", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1234)
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
    text_encoder = get_obj_from_str(config.text_encoder_target)(
        device=device, **config.text_encoder_cfg
    )
    text_encoder.t5.model.to(device)
    scheduler = get_obj_from_str(config.scheduler_target)(**config.val_scheduler_cfg)
    scheduler.num_sampling_steps = args.steps
    scheduler.shift = 7

    model = build_model(args.checkpoint, device)
    wrapper = CFGRegisterWrapper(model)
    examples = choose_examples(
        args.data_root,
        parse_history(args.history_iw_chunks),
        args.samples_per_history,
        args.seed,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "device": args.device,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "examples": [],
    }

    for index, ex in enumerate(examples):
        payload = torch.load(ex["path"], map_location="cpu", weights_only=False)
        chunks = payload["micro_latents"]
        move_all = payload.get("move", torch.zeros(0, dtype=torch.long)).long()
        view_all = payload.get("view", torch.zeros(0, dtype=torch.long)).long()
        sample_dir = args.output_dir / (
            f"{index:03d}_iw{ex['history_iw_chunks']}_"
            f"{ex['path'].parent.name}_{ex['path'].stem}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        registers, local_latent, move, view, target = build_registers(
            model,
            chunks,
            move_all,
            view_all,
            ex["start_micro"],
            ex["history_micro_steps"],
            device=device,
        )
        add = {
            # InfiniteWorld scheduler unconditionally repeats image_cond for CFG.
            # The NAV wrapper ignores it because Register/local_latent are passed
            # explicitly, but the key must exist to reuse the upstream sampler.
            "image_cond": local_latent,
            "registers": registers.repeat(2, 1, 1, 1, 1),
            "local_latent": local_latent.repeat(2, 1, 1, 1, 1),
            "move": move.repeat(2, 1),
            "view": view.repeat(2, 1),
        }
        with torch.inference_mode():
            sample = scheduler.sample(
                model=wrapper,
                text_encoder=text_encoder,
                null_embedder=wrapper.y_embedder,
                z_size=tuple(target.shape),
                prompts=[""],
                guidance_scale=args.cfg_scale,
                negative_prompts=[""],
                device=device,
                additional_args=add,
            )
            gen_px = vae.decode(sample).cpu()
            gt_px = vae.decode(target).cpu()
        save_silent_video(gen_px.to(device), str(sample_dir / "gen"), fps=30, quality=10)
        save_silent_video(gt_px.to(device), str(sample_dir / "gt"), fps=30, quality=10)
        metrics = {
            **{k: str(v) if isinstance(v, Path) else v for k, v in ex.items()},
            "sample_dir": str(sample_dir),
            "gen": video_metrics(gen_px),
            "gt": video_metrics(gt_px),
            "latent_l1": float((sample.float().cpu() - target.float().cpu()).abs().mean()),
        }
        (sample_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
        )
        summary["examples"].append(metrics)
        print(json.dumps({"event": "sample", "index": index, **metrics}, ensure_ascii=False), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary["num_examples"] = len(summary["examples"])
    summary["num_black_like"] = sum(1 for item in summary["examples"] if item["gen"]["black_like"])
    summary["num_static_like"] = sum(1 for item in summary["examples"] if item["gen"]["static_like"])
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
