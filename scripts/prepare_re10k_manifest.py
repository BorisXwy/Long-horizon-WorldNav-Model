#!/usr/bin/env python3
"""把 RE10K 连续帧与相机位姿转换为 NAV 流式训练 manifest。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


MOVE_NAMES = [
    "no-op", "go forward", "go back", "go left", "go right",
    "go forward and go left", "go forward and go right",
    "go back and go left", "go back and go right", "uncertain",
]
VIEW_NAMES = [
    "no-op", "turn up", "turn down", "turn left", "turn right",
    "turn up and turn left", "turn up and turn right",
    "turn down and turn left", "turn down and turn right", "uncertain",
]


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def pose_delta(previous: list[list[float]], current: list[list[float]]):
    p0, p1 = np.asarray(previous), np.asarray(current)
    r0, r1 = nearest_rotation(p0[:, :3]), nearest_rotation(p1[:, :3])
    c0, c1 = -r0.T @ p0[:, 3], -r1.T @ p1[:, 3]
    translation = r0 @ (c1 - c0)
    relative_rotation = r1 @ r0.T
    yaw = math.atan2(relative_rotation[0, 2], relative_rotation[2, 2])
    pitch = math.atan2(
        -relative_rotation[1, 2],
        math.sqrt(relative_rotation[1, 0] ** 2 + relative_rotation[1, 1] ** 2),
    )
    return translation, yaw, pitch


def direction_pair(primary: float, secondary: float, positive: int, negative: int,
                   positive_diag: tuple[int, int], negative_diag: tuple[int, int]) -> int:
    if abs(primary) < 0.414 * abs(secondary):
        return 0
    sign = positive if primary > 0 else negative
    if abs(secondary) < 0.414 * abs(primary):
        return sign
    return positive_diag[0 if secondary > 0 else 1] if primary > 0 else negative_diag[0 if secondary > 0 else 1]


def label_sequence(frames: list[dict]) -> tuple[list[int], list[int], dict]:
    deltas = [pose_delta(a["pose"], b["pose"]) for a, b in zip(frames, frames[1:])]
    magnitudes = np.asarray([np.linalg.norm(x[0][[0, 2]]) for x in deltas])
    angles = np.asarray([math.hypot(x[1], x[2]) for x in deltas])
    trans_scale = max(float(np.median(magnitudes[magnitudes > 1e-8])) if np.any(magnitudes > 1e-8) else 1.0, 1e-8)
    angle_scale = max(float(np.median(angles[angles > 1e-8])) if np.any(angles > 1e-8) else 1.0, 1e-8)

    moves, views = [0], [0]
    for (translation, yaw, pitch), mag, angle in zip(deltas, magnitudes, angles):
        trans_ratio, angle_ratio = mag / trans_scale, angle / angle_scale
        if trans_ratio < 0.25:
            move = 0
        elif trans_ratio > 3.0:
            move = 9
        else:
            # 相机坐标：x 右，-z 前。对角编号遵循 Infinite-World 映射。
            forward, right = -translation[2], translation[0]
            if abs(forward) >= 0.414 * abs(right) and abs(right) >= 0.414 * abs(forward):
                move = 6 if forward > 0 and right > 0 else 5 if forward > 0 else 8 if right > 0 else 7
            elif abs(forward) >= abs(right):
                move = 1 if forward > 0 else 2
            else:
                move = 4 if right > 0 else 3

        if angle_ratio < 0.25:
            view = 0
        elif angle_ratio > 3.0:
            view = 9
        elif abs(yaw) >= 0.414 * abs(pitch) and abs(pitch) >= 0.414 * abs(yaw):
            view = 6 if pitch > 0 and yaw > 0 else 5 if pitch > 0 else 8 if yaw > 0 else 7
        elif abs(yaw) >= abs(pitch):
            view = 4 if yaw > 0 else 3
        else:
            view = 1 if pitch > 0 else 2
        moves.append(move)
        views.append(view)
    return moves, views, {"translation_median": trans_scale, "rotation_median_rad": angle_scale}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/sharedata/RealEstate10K/manifests/frames_train.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("/sharedata/RealEstate10K"))
    parser.add_argument("--output", type=Path, default=Path("/sharedata/RealEstate10K/nav_register/manifests/train.jsonl"))
    parser.add_argument("--frames", type=int, default=162)
    parser.add_argument("--stride", type=int, default=81)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--include-short",
        action="store_true",
        help="将不足目标长度的完整 episode 均匀重采样到目标帧数",
    )
    args = parser.parse_args()

    scenes: dict[str, list[dict]] = defaultdict(list)
    with args.source.open() as stream:
        for line in stream:
            item = json.loads(line)
            scenes[item["scene_id"]].append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    used_scenes = 0
    with args.output.open("w") as output:
        for scene_id, frames in sorted(scenes.items()):
            frames.sort(key=lambda x: x["timestamp"])
            if len(frames) < args.frames:
                if not args.include_short:
                    continue
                indices = np.rint(
                    np.linspace(0, len(frames) - 1, args.frames)
                ).astype(int)
                windows = [
                    (
                        0,
                        [frames[index] for index in indices],
                        True,
                    )
                ]
            else:
                starts = list(
                    range(0, len(frames) - args.frames + 1, args.stride)
                )
                final_start = len(frames) - args.frames
                if final_start not in starts:
                    starts.append(final_start)
                windows = [
                    (start, frames[start : start + args.frames], False)
                    for start in starts
                ]
            used_scenes += 1
            for start, window, resampled in windows:
                move, view, calibration = label_sequence(window)
                suffix = "_resampled" if resampled else ""
                record = {
                    "sample_id": f"{scene_id}_{start:06d}{suffix}",
                    "scene_id": scene_id,
                    "frame_paths": [str(args.root / x["image_path"]) for x in window],
                    "timestamps": [x["timestamp"] for x in window],
                    "move": move,
                    "view": view,
                    "move_names": [MOVE_NAMES[x] for x in move],
                    "view_names": [VIEW_NAMES[x] for x in view],
                    "pose_label_calibration": calibration,
                    "source_num_frames": len(frames),
                    "temporally_resampled": resampled,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            if args.max_scenes and used_scenes >= args.max_scenes:
                break
    print(json.dumps({"output": str(args.output), "samples": count, "scenes": used_scenes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
