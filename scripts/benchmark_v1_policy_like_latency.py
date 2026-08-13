#!/usr/bin/env python3
"""Deprecated benchmark for the removed V1旁路 policy-like path.

当前正式 V1 StageOne checkpoint 里有 action slot/head，但还没有 GigaWorld-style
DiT action-only prefix-cache path。本脚本测的是已经落到 checkpoint 里的路径：

  GT history micro chunks + previous action -> Register
  Register + local latent -> A_query/A_out logits

这能回答“当前留出的 policy token/head 一次前向多久”，但不能代表未来
Stage3 共享 DiT policy-only latency。正式测速请使用
`benchmark_v1_shared_policy_latency.py`。
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
sys.path.insert(0, str(NAV_ROOT / "src"))

from nav.spatial_register_memory import SpatialRegisterMemory
from nav.stage_one_action_interface import StageOneActionInterface


MICRO_FRAMES = 13
MICRO_STRIDE = 12


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
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--history-iw-chunks", type=int, default=1)
    p.add_argument("--start-micro", type=int, default=41)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=500)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    raise SystemExit(
        "benchmark_v1_policy_like_latency.py 已废弃：旧的 Register/local -> "
        "旁路 action head 已从正式模块移除。请使用 "
        "benchmark_v1_shared_policy_latency.py 测 shared Wan/DiT policy path。"
    )
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats(device)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    register_memory = SpatialRegisterMemory(
        channels=16,
        register_frames=4,
        hidden_dim=64,
        num_heads=4,
        num_update_layers=2,
        chunk_time_tokens=4,
    )
    action_interface = StageOneActionInterface()
    register_memory.load_state_dict(state["register_memory"], strict=True)
    action_interface.load_state_dict(state["action_interface"], strict=True)
    register_memory.to(device=device, dtype=torch.bfloat16).eval()
    action_interface.to(device=device, dtype=torch.bfloat16).eval()

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
        conditioned = action_interface.condition_history_chunk(histories[0], moves[0], views[0])
        registers = register_memory.extract(conditioned)
        for h, m, v in zip(histories[1:], moves[1:], views[1:]):
            conditioned = action_interface.condition_history_chunk(h, m, v)
            registers = register_memory.update(registers, conditioned)
        local_latent = histories[-1][:, :, -1:]
    sync(device)
    prep_s = time.perf_counter() - prep_start

    with torch.inference_mode():
        for _ in range(args.warmup):
            out = action_interface.forward_action_query(registers, local_latent)
    sync(device)

    values = []
    with torch.inference_mode():
        for _ in range(args.repeats):
            sync(device)
            start = time.perf_counter()
            out = action_interface.forward_action_query(registers, local_latent)
            sync(device)
            values.append(time.perf_counter() - start)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": state.get("step"),
        "episode": str(args.episode),
        "device": args.device,
        "history_iw_chunks": args.history_iw_chunks,
        "history_micro_steps": history_micro,
        "start_micro": args.start_micro,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "path_semantics": (
            "Current checkpointed StageOneActionInterface only: "
            "Register+local_latent -> A_query/A_out. This is not yet "
            "GigaWorld-style DiT action-only prefix-cache inference."
        ),
        "register_shape": list(registers.shape),
        "local_latent_shape": list(local_latent.shape),
        "move_logits_shape": list(out["move_logits"].shape),
        "view_logits_shape": list(out["view_logits"].shape),
        "history_register_prep_s": prep_s,
        "policy_like_action_head_forward": summarize(values),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None
        ),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
