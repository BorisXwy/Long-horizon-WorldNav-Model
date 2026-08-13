#!/usr/bin/env python3
"""使用 VGGT 为 Kinetics-400 视频标注相机运动；支持全量、分片和断点续跑。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VGGT_ROOT = Path("/sharedata/RealEstate10K/vggt")
sys.path.insert(0, str(VGGT_ROOT))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = b @ a.T
    value = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def camera_center(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    return -(rotation.T @ translation)


def extract_frames(video: Path, count: int, output: Path) -> tuple[list[str], list[float]]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(video),
        ],
        check=True, capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    if duration <= 0:
        raise ValueError("non-positive duration")
    times = np.linspace(0.05 * duration, 0.95 * duration, count)
    paths: list[str] = []
    for index, timestamp in enumerate(times):
        path = output / f"{index:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-ss", f"{timestamp:.6f}",
                "-i", str(video), "-frames:v", "1", "-q:v", "2", "-y", str(path),
            ],
            check=True,
        )
        with Image.open(path) as image:
            image.verify()
        paths.append(str(path))
    return paths, times.tolist()


def choose_videos(root: Path, samples: int, seed: int) -> list[Path]:
    splits = ("train_256", "val_256", "test_256")
    rng = np.random.default_rng(seed)
    allocation = [samples // len(splits)] * len(splits)
    for index in range(samples % len(splits)):
        allocation[index] += 1
    chosen: list[Path] = []
    for split, count in zip(splits, allocation):
        videos = sorted((root / split).glob("*.mp4"))
        indices = rng.choice(len(videos), size=min(count, len(videos)), replace=False)
        chosen.extend(videos[int(index)] for index in indices)
    return chosen


def all_videos(root: Path) -> list[Path]:
    videos: list[Path] = []
    for split in ("train_256", "val_256", "test_256"):
        videos.extend(sorted((root / split).glob("*.mp4")))
    return videos


def stable_shard(path: Path, num_shards: int) -> int:
    digest = hashlib.sha1(str(path).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def classify(rotation_deg: float, translation_depth_ratio: float) -> str:
    if rotation_deg < 2.0 and translation_depth_ratio < 0.02:
        return "low"
    if rotation_deg < 8.0 and translation_depth_ratio < 0.10:
        return "moderate"
    return "strong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/sharedata/datasets/kinetics400"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--all", action="store_true",
        help="处理数据根目录下全部 train/val/test 视频，而不是抽样",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--re10k-calibration", type=Path,
        default=Path(
            "/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/"
            "config/vggt_re10k_confidence.json"
        ),
    )
    parser.add_argument(
        "--weights", type=Path,
        default=Path("/sharedata/RealEstate10K/vggt_weights/model.pt"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index 必须位于 [0, --num-shards)")
    suffix = f".shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    rows_path = args.output / f"camera_motion{suffix}.jsonl"
    videos = all_videos(args.root) if args.all else choose_videos(
        args.root, args.samples, args.seed
    )
    videos = [
        video for video in videos
        if stable_shard(video, args.num_shards) == args.shard_index
    ]
    (args.output / f"episode_list{suffix}.txt").write_text(
        "\n".join(str(path) for path in videos) + "\n"
    )
    calibration = json.loads(args.re10k_calibration.read_text())
    reference = calibration["quality_thresholds"]["reference_like"]
    high = calibration["quality_thresholds"]["high_confidence"]

    device = torch.device(args.device)
    model = VGGT()
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    completed = set()
    if rows_path.exists():
        completed = {
            json.loads(line)["video"]
            for line in rows_path.read_text().splitlines()
            if line.strip()
        }

    with rows_path.open("a") as output:
        for position, video in enumerate(videos, 1):
            if str(video) in completed:
                continue
            row: dict[str, object] = {"video": str(video)}
            try:
                with tempfile.TemporaryDirectory(prefix="kinetics-vggt-") as temp:
                    paths, sample_times = extract_frames(video, args.frames, Path(temp))
                    images = load_and_preprocess_images(paths).to(device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                        tokens, patch_start = model.aggregator(images[None])
                        pose_encoding = model.camera_head(tokens)[-1]
                        extrinsics, _ = pose_encoding_to_extri_intri(
                            pose_encoding, images.shape[-2:]
                        )
                        depth, confidence = model.depth_head(
                            tokens, images[None], patch_start
                        )
                extrinsics_np = extrinsics[0].float().cpu().numpy()
                centers = np.stack([camera_center(item) for item in extrinsics_np])
                rotations = extrinsics_np[:, :, :3]
                end_rotation = rotation_angle_deg(rotations[0], rotations[-1])
                cumulative_rotation = sum(
                    rotation_angle_deg(rotations[i], rotations[i + 1])
                    for i in range(len(rotations) - 1)
                )
                path_length = float(
                    np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()
                )
                depth_np = np.squeeze(depth[0].float().cpu().numpy())
                confidence_np = np.squeeze(confidence[0].float().cpu().numpy())
                conf_p10, conf_p25, conf_p50, conf_p75 = [
                    float(value)
                    for value in np.nanquantile(
                        confidence_np, [0.10, 0.25, 0.50, 0.75]
                    )
                ]
                valid_depth = depth_np[
                    np.isfinite(depth_np)
                    & (depth_np > 0)
                    & (confidence_np > np.nanmedian(confidence_np))
                ]
                median_depth = float(np.median(valid_depth))
                ratio = path_length / max(median_depth, 1e-8)
                if (
                    conf_p25 >= high["confidence_p25_min"]
                    and conf_p50 >= high["confidence_p50_min"]
                ):
                    confidence_level = "high"
                elif (
                    conf_p25 >= reference["confidence_p25_min"]
                    and conf_p50 >= reference["confidence_p50_min"]
                ):
                    confidence_level = "reference"
                else:
                    confidence_level = "low"
                row.update(
                    {
                        "status": "ok",
                        "split": video.parent.name.removesuffix("_256"),
                        "sample_times_sec": sample_times,
                        "end_rotation_deg": end_rotation,
                        "cumulative_rotation_deg": cumulative_rotation,
                        "camera_path": path_length,
                        "median_depth": median_depth,
                        "translation_depth_ratio": ratio,
                        "motion_level": classify(cumulative_rotation, ratio),
                        "confidence_p10": conf_p10,
                        "confidence_p25": conf_p25,
                        "confidence_p50": conf_p50,
                        "confidence_p75": conf_p75,
                        "confidence_level": confidence_level,
                        "motion_label_reliable": confidence_level != "low",
                    }
                )
            except Exception as error:
                row.update({"status": "error", "error": repr(error)})
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{position}/{len(videos)}] {row}", flush=True)

    rows = [
        json.loads(line)
        for line in rows_path.read_text().splitlines()
        if line.strip()
    ]
    valid = [row for row in rows if row.get("status") == "ok"]
    levels = {
        level: sum(row["motion_level"] == level for row in valid)
        for level in ("low", "moderate", "strong")
    }
    summary = {
        "requested": len(videos),
        "completed": len(valid),
        "errors": len(rows) - len(valid),
        "motion_level_counts": levels,
        "motion_level_fractions": {
            key: value / max(len(valid), 1) for key, value in levels.items()
        },
        "median_cumulative_rotation_deg": float(
            np.median([row["cumulative_rotation_deg"] for row in valid])
        ) if valid else None,
        "median_translation_depth_ratio": float(
            np.median([row["translation_depth_ratio"] for row in valid])
        ) if valid else None,
        "thresholds": {
            "low": "rotation < 2 deg and translation/depth < 0.02",
            "moderate": "rotation < 8 deg and translation/depth < 0.10",
            "strong": "otherwise",
        },
    }
    confidence_levels = {
        level: sum(row.get("confidence_level") == level for row in valid)
        for level in ("low", "reference", "high")
    }
    summary.update(
        {
            "all_episodes": args.all,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "confidence_level_counts": confidence_levels,
            "re10k_calibration": str(args.re10k_calibration),
        }
    )
    (args.output / f"summary{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
