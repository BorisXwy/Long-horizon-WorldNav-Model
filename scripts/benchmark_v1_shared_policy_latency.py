#!/usr/bin/env python3
"""Benchmark V1 final shared-backbone policy/action path.

本脚本测的不是旧的旁路小 head，而是正式结构：

  GT history micro chunks + A_hist -> Register
  Register/local/text + A_query prefix tokens -> shared Wan/DiT blocks
  shared action hidden -> action logits

policy-only 前向不输入 future noisy video，也不输入 current action condition。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

NAV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = NAV_ROOT.parent
sys.path.insert(0, str(NAV_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "Infinite-World"))

from nav.infinite_adapter import InfiniteRegisterAdapter


MICRO_FRAMES = 13
MICRO_STRIDE = 12


MODEL_CFG = {
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
}

REGISTER_CFG = {
    "channels": 16,
    "register_frames": 4,
    "hidden_dim": 64,
    "num_heads": 4,
    "num_update_layers": 2,
    "chunk_time_tokens": 4,
}


def micro_steps_for_iw_chunks(iw_chunks: int) -> int:
    target_frames = 1 + 80 * iw_chunks
    if target_frames <= MICRO_FRAMES:
        return 1
    return math.ceil((target_frames - MICRO_FRAMES) / MICRO_STRIDE) + 1


def pad_action_81(values: torch.Tensor, start: int, length: int) -> torch.Tensor:
    out = torch.zeros(81, dtype=torch.long)
    if values.numel() == 0:
        return out
    segment = values[start : min(start + length, values.numel())].long()
    if segment.numel():
        out[-segment.numel() :] = segment
    return out


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device.index or 0)


def summarize(values: list[float]) -> dict:
    return {
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
        "values_s": values[:20],
        "num_values": len(values),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True, help="NAV checkpoint; old checkpoints are loaded strict=False for new action-token params")
    p.add_argument("--base-checkpoint", type=Path, default=Path("/sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"))
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--text-embedding", type=Path, default=Path("/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--history-iw-chunks", type=int, default=1)
    p.add_argument("--start-micro", type=int, default=41)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_text(path: Path, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    y = payload["y"].to(device=device, dtype=torch.bfloat16)
    y_mask = payload["y_mask"].to(device=device)
    return y, y_mask


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[InfiniteRegisterAdapter, dict]:
    model = InfiniteRegisterAdapter(
        variant="latent_prefix",
        model_cfg=MODEL_CFG,
        register_cfg=REGISTER_CFG,
    )
    base_audit = model.load_infinite_checkpoint(str(args.base_checkpoint))
    checkpoint_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    current_backbone = model.backbone.state_dict()
    raw_backbone = checkpoint_state.get("backbone", {})
    filtered_backbone = {
        key: value
        for key, value in raw_backbone.items()
        if key in current_backbone and tuple(value.shape) == tuple(current_backbone[key].shape)
    }
    mismatched_backbone = [
        key
        for key, value in raw_backbone.items()
        if key in current_backbone and tuple(value.shape) != tuple(current_backbone[key].shape)
    ]
    backbone_result = model.backbone.load_state_dict(
        filtered_backbone,
        strict=False,
    )
    register_result = model.register_memory.load_state_dict(
        checkpoint_state.get("register_memory", {}),
        strict=True,
    )
    action_interface_result = model.action_interface.load_state_dict(
        checkpoint_state.get("action_interface", {}),
        strict=False,
    )
    shared_action_result = model.load_shared_action_state_dict(
        checkpoint_state.get("shared_action", {})
    )
    model.to(device=device, dtype=torch.bfloat16).eval()
    return model, {
        "base_audit": base_audit,
        "checkpoint_step": checkpoint_state.get("step"),
        "checkpoint_variant": checkpoint_state.get("variant"),
        "backbone_missing_keys": list(backbone_result.missing_keys),
        "backbone_unexpected_keys": list(backbone_result.unexpected_keys),
        "backbone_mismatched_keys": mismatched_backbone,
        "register_missing_keys": list(register_result.missing_keys),
        "register_unexpected_keys": list(register_result.unexpected_keys),
        "action_interface_missing_keys": list(action_interface_result.missing_keys),
        "action_interface_unexpected_keys": list(action_interface_result.unexpected_keys),
        "shared_action_missing_keys": shared_action_result["missing_keys"],
        "shared_action_unexpected_keys": shared_action_result["unexpected_keys"],
        "new_shared_action_params": (
            "current_action/action_query/action_head are newly initialized when "
            "loading old pre-shared-action-token checkpoints"
        ),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats(device)

    model, load_audit = load_model(args, device)
    y, y_mask = load_text(args.text_embedding, device=device)

    payload = torch.load(args.episode, map_location="cpu", weights_only=False)
    chunks = payload["micro_latents"]
    move_all = payload.get("move", torch.zeros(0, dtype=torch.long)).long()
    view_all = payload.get("view", torch.zeros(0, dtype=torch.long)).long()
    history_micro = micro_steps_for_iw_chunks(args.history_iw_chunks)

    histories = []
    moves = []
    views = []
    for offset in range(history_micro):
        micro_index = args.start_micro + offset
        histories.append(chunks[:, micro_index].to(device=device, dtype=torch.bfloat16))
        frame_start = micro_index * MICRO_STRIDE
        moves.append(pad_action_81(move_all, frame_start, MICRO_FRAMES).to(device)[None])
        views.append(pad_action_81(view_all, frame_start, MICRO_FRAMES).to(device)[None])

    sync(device)
    prep_start = time.perf_counter()
    with torch.inference_mode():
        conditioned = model.action_interface.condition_history_chunk(histories[0], moves[0], views[0])
        registers = model.register_memory.extract(conditioned)
        for h, m, v in zip(histories[1:], moves[1:], views[1:]):
            conditioned = model.action_interface.condition_history_chunk(h, m, v)
            registers = model.register_memory.update(registers, conditioned)
        local_latent = histories[-1][:, :, -1:]
    sync(device)
    prep_s = time.perf_counter() - prep_start

    with torch.inference_mode():
        for _ in range(args.warmup):
            out = model.policy_forward(
                y=y,
                y_mask=y_mask,
                registers=registers,
                local_latent=local_latent,
            )
    sync(device)

    values = []
    with torch.inference_mode():
        for _ in range(args.repeats):
            sync(device)
            start = time.perf_counter()
            out = model.policy_forward(
                y=y,
                y_mask=y_mask,
                registers=registers,
                local_latent=local_latent,
            )
            sync(device)
            values.append(time.perf_counter() - start)

    result = {
        "checkpoint": str(args.checkpoint),
        "base_checkpoint": str(args.base_checkpoint),
        "episode": str(args.episode),
        "device": args.device,
        "history_iw_chunks": args.history_iw_chunks,
        "history_micro_steps": history_micro,
        "start_micro": args.start_micro,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "path_semantics": (
            "V1 final shared-backbone policy path: Register/local/text + A_query "
            "prefix tokens run through Wan/DiT blocks; no future noisy video and "
            "no current action condition are provided."
        ),
        "register_shape": list(registers.shape),
        "local_latent_shape": list(local_latent.shape),
        "move_logits_shape": list(out["move_logits"].shape),
        "view_logits_shape": list(out["view_logits"].shape),
        "history_register_prep_s": prep_s,
        "shared_policy_forward": summarize(values),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None
        ),
        "load_audit": load_audit,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
