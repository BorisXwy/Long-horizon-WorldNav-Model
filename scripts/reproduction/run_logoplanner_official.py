#!/usr/bin/env python3
"""使用官方 LoGoPlanner 权重在一组真实 RGB-D 输入上执行完整策略前向。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=12)
    return parser.parse_args()


def load_rgbd(rgb_dir: Path, depth_dir: Path, count: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rgb_paths = sorted(rgb_dir.glob("*.png"))[:count]
    if len(rgb_paths) < count:
        raise ValueError(f"需要 {count} 张 RGB，实际只有 {len(rgb_paths)} 张")
    rgbs, depths, names = [], [], []
    for rgb_path in rgb_paths:
        depth_path = depth_dir / rgb_path.name
        if not depth_path.exists():
            depth_path = depth_dir / f"{rgb_path.stem}.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"找不到与 {rgb_path.name} 配对的 depth")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = (
            np.asarray(np.load(depth_path), dtype=np.float32)
            if depth_path.suffix == ".npy"
            else np.asarray(Image.open(depth_path), dtype=np.float32)
        )
        if np.nanmedian(depth[depth > 0]) > 20:
            depth = depth / 1000.0
        rgb = cv2.resize(rgb, (308, 168), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        depth = cv2.resize(depth, (308, 168), interpolation=cv2.INTER_NEAREST)[..., None]
        depth[(depth > 5.0) | (depth < 0.1) | ~np.isfinite(depth)] = 0
        rgbs.append(rgb)
        depths.append(depth)
        names.append(rgb_path.name)
    return np.stack(rgbs), np.stack(depths), names


def main() -> None:
    args = parse_args()
    if args.num_frames != 12:
        raise ValueError("官方发布权重的 geometry context 固定为 12 帧")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logo_root = args.repo / "baselines" / "logoplanner"
    sys.path.insert(0, str(logo_root))

    from policy_network import LoGoPlanner_Policy

    rgbs, depths, names = load_rgbd(args.rgb_dir, args.depth_dir, args.num_frames)
    context_rgbd = np.concatenate([rgbs, depths], axis=-1)[None]
    memory_rgbd = context_rgbd[:, -8:]
    start_goal = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32)

    model = LoGoPlanner_Policy(
        image_size=224,
        memory_size=8,
        context_size=12,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        device="cuda:0",
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    incompatible = model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = model.predict_pointgoal_action(start_goal, memory_rgbd, context_rgbd, sample_num=16)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    all_trajectory, critic, positive, negative, sub_goal = outputs

    np.savez_compressed(
        args.output_dir / "policy_outputs.npz",
        all_trajectory=all_trajectory,
        critic_values=critic,
        positive_trajectory=positive,
        negative_trajectory=negative,
        sub_pointgoal=sub_goal,
    )
    metrics = {
        "model": "LoGoPlanner",
        "repo_commit": subprocess.check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "pi3_commit": subprocess.check_output(
            ["git", "-C", str(logo_root / "Pi3"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch_version": torch.__version__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "component_parameters": {
            "state_encoder": sum(parameter.numel() for parameter in model.state_encoder.parameters()),
            "rgbd_encoder": sum(parameter.numel() for parameter in model.rgbd_encoder.parameters()),
            "policy_transformer_decoder": sum(
                parameter.numel() for parameter in model.decoder.parameters()
            ),
            "action_head": sum(parameter.numel() for parameter in model.action_head.parameters()),
            "critic_head": sum(parameter.numel() for parameter in model.critic_head.parameters()),
        },
        "checkpoint": str(args.checkpoint),
        "input": "NYUDv2 paired RGB-D (functional reproduction; not benchmark sequence)",
        "input_frames": names,
        "context_frames": 12,
        "recent_memory_frames": 8,
        "trajectory_horizon": 24,
        "trajectory_candidates": 16,
        "diffusion_steps": 10,
        "inference_seconds": elapsed,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "output_shapes": {
            "all_trajectory": list(all_trajectory.shape),
            "critic_values": list(critic.shape),
            "positive_trajectory": list(positive.shape),
            "negative_trajectory": list(negative.shape),
            "sub_pointgoal": list(sub_goal.shape),
        },
        "finite": {
            "all_trajectory": bool(np.isfinite(all_trajectory).all()),
            "critic_values": bool(np.isfinite(critic).all()),
            "sub_pointgoal": bool(np.isfinite(sub_goal).all()),
        },
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
