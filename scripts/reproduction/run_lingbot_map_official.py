#!/usr/bin/env python3
"""在真实图像序列上运行官方 LingBot-Map 流式前向并保存结构化结果。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-k", type=int, default=24)
    parser.add_argument("--scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    return parser.parse_args()


def tensor_summary(value: torch.Tensor) -> dict[str, object]:
    value = value.detach().float().cpu()
    finite = torch.isfinite(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite_ratio": float(finite.float().mean()),
        "min": float(value[finite].min()) if finite.any() else None,
        "max": float(value[finite].max()) if finite.any() else None,
        "mean": float(value[finite].mean()) if finite.any() else None,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))

    from demo import load_images, load_model, postprocess

    device = torch.device("cuda")
    model_args = SimpleNamespace(
        mode="streaming",
        image_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        num_scale_frames=args.scale_frames,
        use_sdpa=True,
        camera_num_iterations=4,
        model_path=str(args.checkpoint),
    )

    images, paths, _ = load_images(
        image_folder=str(args.image_dir),
        first_k=args.first_k,
        image_size=518,
        patch_size=14,
    )
    if images.shape[0] <= args.scale_frames:
        raise ValueError("图像数量必须大于 scale frames，才能覆盖逐帧流式阶段")

    load_start = time.perf_counter()
    model = load_model(model_args, device)
    model.aggregator = model.aggregator.to(dtype=torch.bfloat16)
    load_seconds = time.perf_counter() - load_start
    images = images.to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    infer_start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=args.scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=None,
        )
    torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - infer_start
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    cache_info = model.get_kv_cache_info()

    predictions, images_cpu = postprocess(predictions, images)
    save_payload = {
        key: value.detach().cpu()
        for key, value in predictions.items()
        if torch.is_tensor(value) and key != "images"
    }
    save_payload["images"] = images_cpu.detach().cpu()
    torch.save(save_payload, args.output_dir / "predictions.pt")

    summaries = {
        key: tensor_summary(value)
        for key, value in save_payload.items()
        if torch.is_tensor(value)
    }
    metrics = {
        "model": "LingBot-Map",
        "repo_commit": None,
        "torch_version": torch.__version__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(args.checkpoint),
        "input_dir": str(args.image_dir),
        "input_paths": [str(path) for path in paths],
        "num_frames": int(images.shape[0]),
        "scale_frames": args.scale_frames,
        "stream_frames": int(images.shape[0] - args.scale_frames),
        "keyframe_interval": args.keyframe_interval,
        "attention_backend": "sdpa",
        "load_seconds": load_seconds,
        "inference_seconds": infer_seconds,
        "fps_including_scale_phase": float(images.shape[0] / infer_seconds),
        "gpu_peak_allocated_gib": peak_gib,
        "kv_cache_info_after_inference": cache_info,
        "outputs": summaries,
    }
    import subprocess

    metrics["repo_commit"] = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
