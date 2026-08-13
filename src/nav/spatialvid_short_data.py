"""SpatialVID short-history 数据选择与稀疏 Pose 对齐工具。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


CHUNK_FRAMES = 81


def annotation_paths(video: Path) -> tuple[Path, Path]:
    base = video.parents[2]
    annotation = base / "annotations" / video.parent.name / video.stem
    return annotation / "poses.npy", annotation / "indexes.txt"


def load_sparse_annotation(video: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    pose_path, indexes_path = annotation_paths(video)
    poses = np.load(pose_path)
    indexes = np.loadtxt(indexes_path, dtype=np.int64)
    if indexes.ndim == 1:
        indexes = indexes[None]
    if indexes.shape[1] < 2:
        raise ValueError(f"indexes.txt 缺少原视频帧号列：{indexes_path}")

    raw_pose_count = len(poses)
    raw_index_count = len(indexes)
    paired = min(raw_pose_count, raw_index_count)
    if paired < 2:
        raise ValueError(f"有效 Pose 对不足 2：{video}")
    poses = poses[:paired]
    frame_indexes = indexes[:paired, 1]

    order = np.argsort(frame_indexes, kind="stable")
    poses = poses[order]
    frame_indexes = frame_indexes[order]
    unique = np.concatenate(([True], np.diff(frame_indexes) > 0))
    poses = poses[unique]
    frame_indexes = frame_indexes[unique]
    if len(frame_indexes) < 2:
        raise ValueError(f"去重后有效 Pose 对不足 2：{video}")

    return poses, frame_indexes, {
        "raw_pose_count": raw_pose_count,
        "raw_index_count": raw_index_count,
        "paired_pose_count": paired,
        "unique_pose_count": len(frame_indexes),
        "length_mismatch": raw_pose_count != raw_index_count,
    }


def source_frame_count(video: Path) -> int:
    """使用 indexes 末项；SpatialVID 官方定义其为原视频最后一帧。"""

    _, indexes_path = annotation_paths(video)
    indexes = np.loadtxt(indexes_path, dtype=np.int64)
    if indexes.ndim == 1:
        indexes = indexes[None]
    return int(np.max(indexes[:, 1])) + 1


def select_short_window(
    video: Path, *, max_chunks: int = 3
) -> tuple[np.ndarray, dict]:
    frame_count = source_frame_count(video)
    source_chunks = frame_count // CHUNK_FRAMES
    if source_chunks < 2:
        raise ValueError(f"不足两个 chunk：{video}")
    cached_chunks = min(max_chunks, source_chunks)
    possible_starts = source_chunks - cached_chunks + 1
    stable_value = int.from_bytes(
        hashlib.sha256(video.stem.encode()).digest()[:8], "big"
    )
    start_chunk = stable_value % possible_starts
    start_frame = start_chunk * CHUNK_FRAMES
    frame_indices = np.arange(
        start_frame,
        start_frame + cached_chunks * CHUNK_FRAMES,
        dtype=np.int64,
    )
    return frame_indices, {
        "source_num_frames": frame_count,
        "source_num_chunks": source_chunks,
        "window_start_chunk": start_chunk,
        "cached_chunks": cached_chunks,
        "chunk_frames": CHUNK_FRAMES,
    }


def interpolate_poses(
    sparse_pose: np.ndarray,
    sparse_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    if sparse_pose.ndim != 2 or sparse_pose.shape[1] < 7:
        raise ValueError(f"未知 SpatialVID Pose 形状：{sparse_pose.shape}")
    rotations = Rotation.from_quat(sparse_pose[:, 3:7])
    target = np.clip(target_indices, sparse_indices[0], sparse_indices[-1])
    out_rotation = Slerp(sparse_indices, rotations)(target).as_matrix()
    out_translation = np.stack(
        [
            np.interp(target, sparse_indices, sparse_pose[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )
    return np.concatenate(
        [out_rotation, out_translation[:, :, None]], axis=2
    )


def iter_spatialvid_videos(root: Path):
    for group in sorted((root / "videos").glob("group_*")):
        for video in sorted(group.glob("*.mp4")):
            pose_path, indexes_path = annotation_paths(video)
            if pose_path.is_file() and indexes_path.is_file():
                yield video
