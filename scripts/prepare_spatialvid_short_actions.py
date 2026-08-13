#!/usr/bin/env python3
"""为 SpatialVID 的确定性 2–3 chunk 窗口生成逐帧 action。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nav.spatialvid_short_data import (
    interpolate_poses,
    iter_spatialvid_videos,
    load_sparse_annotation,
    select_short_window,
)

# 复用统一数据流水线中已经审计过的 action 离散化定义。
from prepare_multidataset_manifest import labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/sharedata/datasets/SpatialVID/raw/huggingface/SpatialVID"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/sharedata/NAV/derived/actions/spatialvid_short"),
    )
    parser.add_argument("--max-chunks", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    videos = iter_spatialvid_videos(args.root)
    if args.max_episodes:
        videos = islice(videos, args.max_episodes)
    for position, video in enumerate(videos, 1):
        sample_id = f"spatialvid__{video.stem}"
        output = args.output / f"{sample_id}.json"
        if output.is_file():
            counts["existing"] += 1
            continue
        try:
            frame_indices, window = select_short_window(
                video, max_chunks=args.max_chunks
            )
            sparse_pose, sparse_indices, audit = load_sparse_annotation(video)
            # 先在完整可用 episode 上定义动作尺度和相邻帧变换，再切窗口。
            # 这样随机窗口首帧保留与前一帧的真实边界动作，也与旧统一
            # manifest 的逐 episode calibration 语义一致。
            full_indices = np.arange(
                window["source_num_chunks"] * window["chunk_frames"],
                dtype=np.int64,
            )
            full_poses = interpolate_poses(
                sparse_pose, sparse_indices, full_indices
            )
            full_move, full_view, calibration = labels(full_poses)
            begin = int(frame_indices[0])
            end = begin + len(frame_indices)
            move = full_move[begin:end]
            view = full_view[begin:end]
            payload = {
                "sample_id": sample_id,
                "dataset": "spatialvid",
                "video_path": str(video),
                "frame_indices": frame_indices.tolist(),
                "move": move,
                "view": view,
                "pose_label_calibration": calibration,
                **window,
                "annotation_audit": audit,
            }
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
            temporary.replace(output)
            counts["completed"] += 1
            counts["length_mismatch"] += int(audit["length_mismatch"])
        except Exception as error:
            counts["errors"] += 1
            print(json.dumps({
                "position": position,
                "error": sample_id,
                "message": repr(error),
            }, ensure_ascii=False), flush=True)
        if position % 100 == 0:
            print(json.dumps({
                "position": position, **counts
            }, ensure_ascii=False), flush=True)
    summary = {"output": str(args.output), **counts}
    (args.output / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "_COMPLETE").touch()
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
