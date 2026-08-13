#!/usr/bin/env python3
"""从 canonical manifest 原子导出每个完整 episode 的 dense Action。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    counts = Counter()
    for line in args.manifest.open(encoding="utf-8"):
        item = json.loads(line)
        output = (
            args.output / item["dataset"] / f'{item["sample_id"]}.json'
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "nav_dense_action_v1",
            "sample_id": item["sample_id"],
            "dataset": item["dataset"],
            "episode_id": item["episode_id"],
            "num_frames": item["num_frames"],
            "num_chunks": item["num_chunks"],
            "chunk_frames": item["chunk_frames"],
            "frame_indices": item["frame_indices"],
            "move": item["move"],
            "view": item["view"],
            "pose_label_calibration": item[
                "pose_label_calibration"
            ],
            "pose_time_mapping": item.get("pose_time_mapping"),
            "sparse_pose_frames": item.get("sparse_pose_frames"),
        }
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        counts[item["dataset"]] += 1

    summary = {
        "schema": "nav_dense_action_v1",
        "manifest": str(args.manifest),
        "output": str(args.output),
        "episodes": sum(counts.values()),
        "datasets": dict(counts),
    }
    (args.output / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "_COMPLETE").touch()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
