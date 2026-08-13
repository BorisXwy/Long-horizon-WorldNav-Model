#!/usr/bin/env python3
"""扫描只读下载目录，重建可增量复用的 NAV 多数据集 episode manifest。"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

def invert_pose(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)[:3].astype(np.float64)


def labels(poses: np.ndarray) -> tuple[list[int], list[int], dict]:
    rotations = poses[:, :, :3]
    translations = poses[:, :, 3]
    centers = -np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), translations)
    relative_translation = np.einsum(
        "nij,nj->ni", rotations[:-1], np.diff(centers, axis=0)
    )
    relative_rotation = rotations[1:] @ rotations[:-1].transpose(0, 2, 1)
    yaw = np.arctan2(relative_rotation[:, 0, 2], relative_rotation[:, 2, 2])
    pitch = np.arctan2(
        -relative_rotation[:, 1, 2],
        np.sqrt(
            relative_rotation[:, 1, 0] ** 2
            + relative_rotation[:, 1, 1] ** 2
        ),
    )
    magnitudes = np.linalg.norm(relative_translation[:, [0, 2]], axis=1)
    angles = np.hypot(yaw, pitch)
    trans_scale = max(
        float(np.median(magnitudes[magnitudes > 1e-8]))
        if np.any(magnitudes > 1e-8) else 1.0,
        1e-8,
    )
    angle_scale = max(
        float(np.median(angles[angles > 1e-8]))
        if np.any(angles > 1e-8) else 1.0,
        1e-8,
    )
    moves, views = [0], [0]
    for translation, y, p, magnitude, angle in zip(
        relative_translation, yaw, pitch, magnitudes, angles
    ):
        trans_ratio, angle_ratio = magnitude / trans_scale, angle / angle_scale
        if trans_ratio < 0.25:
            move = 0
        elif trans_ratio > 3.0:
            move = 9
        else:
            forward, right = -translation[2], translation[0]
            if (
                abs(forward) >= 0.414 * abs(right)
                and abs(right) >= 0.414 * abs(forward)
            ):
                move = (
                    6 if forward > 0 and right > 0
                    else 5 if forward > 0
                    else 8 if right > 0
                    else 7
                )
            elif abs(forward) >= abs(right):
                move = 1 if forward > 0 else 2
            else:
                move = 4 if right > 0 else 3
        if angle_ratio < 0.25:
            view = 0
        elif angle_ratio > 3.0:
            view = 9
        elif abs(y) >= 0.414 * abs(p) and abs(p) >= 0.414 * abs(y):
            view = (
                6 if p > 0 and y > 0
                else 5 if p > 0
                else 8 if y > 0
                else 7
            )
        elif abs(y) >= abs(p):
            view = 4 if y > 0 else 3
        else:
            view = 1 if p > 0 else 2
        moves.append(move)
        views.append(view)
    return moves, views, {
        "translation_median": trans_scale,
        "rotation_median_rad": angle_scale,
    }


def trim_indices(length: int, chunk_frames: int) -> np.ndarray:
    usable = length // chunk_frames * chunk_frames
    return np.arange(usable, dtype=np.int64)


def record(
    dataset: str,
    episode_id: str,
    source: dict,
    frame_indices: np.ndarray,
    poses: np.ndarray,
    chunk_frames: int,
) -> dict | None:
    if len(frame_indices) < 2 * chunk_frames:
        return None
    move, view, calibration = labels(poses)
    return {
        "sample_id": f"{dataset}__{episode_id}",
        "dataset": dataset,
        "episode_id": episode_id,
        **source,
        "frame_indices": frame_indices.tolist(),
        "move": move,
        "view": view,
        "num_frames": len(frame_indices),
        "num_chunks": len(frame_indices) // chunk_frames,
        "chunk_frames": chunk_frames,
        "pose_label_calibration": calibration,
    }


def scan_re10k(root: Path, chunk_frames: int):
    manifest = root / "nav_register/manifests/train_all.jsonl"
    if not manifest.is_file():
        return
    for line in manifest.open():
        item = json.loads(line)
        usable = len(item["frame_paths"]) // chunk_frames * chunk_frames
        if usable < 2 * chunk_frames:
            continue
        yield {
            "sample_id": f're10k__{item["sample_id"]}',
            "dataset": "re10k",
            "episode_id": item["sample_id"],
            "source_type": "images",
            "frame_paths": item["frame_paths"][:usable],
            "frame_indices": list(range(usable)),
            "move": item["move"][:usable],
            "view": item["view"][:usable],
            "num_frames": usable,
            "num_chunks": usable // chunk_frames,
            "chunk_frames": chunk_frames,
            "pose_label_calibration": item["pose_label_calibration"],
        }


def scan_dl3dv(root: Path, chunk_frames: int):
    manifest = root / "manifests/budget_100gb_selection.tsv"
    if not manifest.is_file():
        return
    for line in manifest.read_text().splitlines()[1:]:
        fields = line.split("\t")
        scene = fields[1]
        scene_root = root / "raw/480p" / fields[0] / scene
        transform_path = scene_root / "transforms.json"
        if not transform_path.is_file():
            continue
        data = json.loads(transform_path.read_text())
        frames = data["frames"]
        video_candidates = list(
            (root / "raw/videos").glob(f"*/{scene}/video.mp4")
        )
        if len(video_candidates) != 1:
            continue
        video_path = video_candidates[0]
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        indices = trim_indices(frame_count, chunk_frames)
        if len(indices) < 2 * chunk_frames:
            continue
        sparse_poses = np.stack(
            [
                invert_pose(np.asarray(frame["transform_matrix"]))
                for frame in frames
            ]
        )
        # 官方 transforms.json 只包含稀疏 SfM 帧，文件中没有原视频 frame id。
        # DL3DV 的抽帧覆盖完整 clip，因此按首尾对齐的均匀时间轴映射，再在
        # world-to-camera 外参空间进行 rotation Slerp / translation linear。
        sparse_indices = np.linspace(
            0, frame_count - 1, len(sparse_poses), dtype=np.float64
        )
        poses = interpolate_matrix_pose(
            sparse_poses, sparse_indices, indices
        )
        item = record(
            "dl3dv", scene,
            {
                "source_type": "video",
                "video_path": str(video_path),
                "pose_time_mapping": "uniform_full_clip_v1",
                "sparse_pose_frames": len(sparse_poses),
            },
            indices, poses, chunk_frames,
        )
        if item:
            yield item


def interpolate_spatial_pose(
    sparse_pose: np.ndarray, sparse_indices: np.ndarray, target: np.ndarray
) -> np.ndarray:
    rotations = Rotation.from_quat(sparse_pose[:, 3:7])
    target_clipped = np.clip(target, sparse_indices[0], sparse_indices[-1])
    out_rotation = Slerp(sparse_indices, rotations)(target_clipped).as_matrix()
    out_translation = np.stack(
        [
            np.interp(target_clipped, sparse_indices, sparse_pose[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )
    return np.concatenate([out_rotation, out_translation[:, :, None]], axis=2)


def interpolate_matrix_pose(
    sparse_pose: np.ndarray,
    sparse_indices: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """把稀疏 world-to-camera 外参插值到连续视频帧。"""

    target_clipped = np.clip(target, sparse_indices[0], sparse_indices[-1])
    rotations = Rotation.from_matrix(sparse_pose[:, :, :3])
    out_rotation = Slerp(sparse_indices, rotations)(
        target_clipped
    ).as_matrix()
    out_translation = np.stack(
        [
            np.interp(
                target_clipped, sparse_indices, sparse_pose[:, axis, 3]
            )
            for axis in range(3)
        ],
        axis=1,
    )
    return np.concatenate(
        [out_rotation, out_translation[:, :, None]], axis=2
    )


def scan_spatialvid(root: Path, chunk_frames: int):
    base = root / "raw/huggingface/SpatialVID"
    for group in sorted((base / "videos").glob("group_*")):
        annotations = base / "annotations" / group.name
        if not annotations.is_dir():
            continue
        for video in sorted(group.glob("*.mp4")):
            annotation = annotations / video.stem
            pose_path = annotation / "poses.npy"
            indexes_path = annotation / "indexes.txt"
            if not pose_path.is_file() or not indexes_path.is_file():
                continue
            sparse_pose = np.load(pose_path)
            raw_indexes = np.loadtxt(indexes_path, dtype=np.int64)
            if raw_indexes.ndim == 1:
                raw_indexes = raw_indexes[None]
            if raw_indexes.shape[1] < 2:
                continue
            # 官方少量条目的 Pose 与 index 数量不一致。按共同前缀配对，
            # 再按原视频帧号稳定排序、去重；与已验证 short pipeline 一致。
            paired = min(len(sparse_pose), len(raw_indexes))
            if paired < 2:
                continue
            sparse_pose = sparse_pose[:paired]
            sparse_indices = raw_indexes[:paired, 1]
            order = np.argsort(sparse_indices, kind="stable")
            sparse_pose = sparse_pose[order]
            sparse_indices = sparse_indices[order]
            unique = np.concatenate(
                ([True], np.diff(sparse_indices) > 0)
            )
            sparse_pose = sparse_pose[unique]
            sparse_indices = sparse_indices[unique]
            if len(sparse_indices) < 2:
                continue
            # SpatialVID 的 indexes 是原视频帧号，末项对应最后一帧。
            # 使用它避免每次增量扫描都打开 5,000 个远端 MP4。
            frame_count = int(sparse_indices[-1]) + 1
            indices = trim_indices(frame_count, chunk_frames)
            poses = interpolate_spatial_pose(sparse_pose, sparse_indices, indices)
            item = record(
                "spatialvid", video.stem,
                {"source_type": "video", "video_path": str(video)},
                indices, poses, chunk_frames,
            )
            if item:
                yield item


def quaternion_matrix(rows: pd.DataFrame) -> np.ndarray:
    rotation = Rotation.from_quat(
        rows[["qx", "qy", "qz", "qw"]].to_numpy()
    ).as_matrix()
    translation = rows[["tx_m", "ty_m", "tz_m"]].to_numpy()
    output = np.repeat(np.eye(4)[None], len(rows), axis=0)
    output[:, :3, :3] = rotation
    output[:, :3, 3] = translation
    return output


def scan_argoverse(root: Path, chunk_frames: int):
    for episode in sorted((root / "train").glob("*")):
        if not (episode / ".download_complete").is_file():
            continue
        camera_dir = episode / "sensors/cameras/ring_front_center"
        images = sorted(camera_dir.glob("*.jpg"))
        if len(images) < 2 * chunk_frames:
            continue
        usable = len(images) // chunk_frames * chunk_frames
        selected = np.rint(np.linspace(0, len(images) - 1, usable)).astype(int)
        image_paths = [str(images[index]) for index in selected]
        image_ts = np.asarray([int(images[index].stem) for index in selected])

        ego = pd.read_feather(episode / "city_SE3_egovehicle.feather")
        ego_ts = ego["timestamp_ns"].to_numpy()
        nearest = np.searchsorted(ego_ts, image_ts)
        nearest = np.clip(nearest, 1, len(ego_ts) - 1)
        left = nearest - 1
        nearest -= (
            np.abs(ego_ts[left] - image_ts)
            < np.abs(ego_ts[nearest] - image_ts)
        )
        city_from_ego = quaternion_matrix(ego.iloc[nearest])
        calibration = pd.read_feather(
            episode / "calibration/egovehicle_SE3_sensor.feather"
        )
        sensor = calibration[
            calibration["sensor_name"] == "ring_front_center"
        ]
        ego_from_camera = quaternion_matrix(sensor)[0]
        city_from_camera = city_from_ego @ ego_from_camera
        poses = np.stack([invert_pose(matrix) for matrix in city_from_camera])
        item = record(
            "argoverse2", episode.name,
            {"source_type": "images", "frame_paths": image_paths},
            np.arange(usable), poses, chunk_frames,
        )
        if item:
            yield item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("/sharedata/NAV/derived/manifests/episodes.jsonl"),
    )
    parser.add_argument("--chunk-frames", type=int, default=81)
    parser.add_argument(
        "--datasets",
        default="re10k,dl3dv,spatialvid,argoverse2",
        help="逗号分隔：re10k,dl3dv,spatialvid,argoverse2",
    )
    args = parser.parse_args()

    requested_names = [
        name.strip() for name in args.datasets.split(",") if name.strip()
    ]
    requested = set(requested_names)
    available = {
        "re10k": scan_re10k(
            Path("/sharedata/RealEstate10K"), args.chunk_frames
        ),
        "dl3dv": scan_dl3dv(
            Path("/sharedata/datasets/DL3DV-10K"), args.chunk_frames
        ),
        "spatialvid": scan_spatialvid(
            Path("/sharedata/datasets/SpatialVID"), args.chunk_frames
        ),
        "argoverse2": scan_argoverse(
            Path("/sharedata/datasets/Argoverse2-Sensor-100GB"),
            args.chunk_frames,
        ),
    }
    unknown = requested - available.keys()
    if unknown:
        raise ValueError(f"未知数据集：{sorted(unknown)}")
    # 严格保留命令行给出的数据集顺序。cache worker按 manifest 顺序处理，
    # 因此可以优先完成某个基础数据集，再继续全量后台准备。
    scanners = [available[name] for name in requested_names]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    counts: Counter = Counter()
    chunk_counts: Counter = Counter()
    with temporary.open("w") as stream:
        for scanner in scanners:
            for item in scanner:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[item["dataset"]] += 1
                chunk_counts[item["num_chunks"]] += 1
    temporary.replace(args.output)
    report = {
        "manifest": str(args.output),
        "episodes": sum(counts.values()),
        "datasets": dict(counts),
        "chunk_distribution": dict(sorted(chunk_counts.items())),
        "download_directories_modified": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
