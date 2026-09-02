#!/usr/bin/env python3
"""为 StreamVLN 开环评测建立 R2R 渲染数据的轻量适配视图。

NAV 的 R2R 渲染结果已经包含 PNG 帧、instruction 和 action 元数据，
但目录布局与 StreamVLN 官方轨迹包不同。本脚本只创建符号链接和小型
annotations.json，不复制图像，也不修改原始渲染目录。
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def _read_gt_actions(item: dict[str, Any]) -> list[int]:
    action_path = Path(item["episode_action_path"])
    payload = json.loads(action_path.read_text())
    values = payload.get("gt_actions")
    if values is None:
        raise KeyError(f"gt_actions missing from {action_path}")
    return [int(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--max-episodes", type=int, default=-1)
    args = parser.parse_args()

    source_jsonl = args.render_root / "episodes" / "rendered_episodes.jsonl.gz"
    if not source_jsonl.exists():
        raise FileNotFoundError(source_jsonl)

    args.output_root.mkdir(parents=True, exist_ok=True)
    image_root = args.output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    annotations: list[dict[str, Any]] = []
    with gzip.open(source_jsonl, "rt") as handle:
        for line in handle:
            item = json.loads(line)
            if args.split and item.get("split") != args.split:
                continue
            frame_dir = Path(item["render_frame_dir"])
            if not frame_dir.is_dir():
                raise FileNotFoundError(frame_dir)
            video_id = str(item["sample_id"])
            video_dir = image_root / video_id
            rgb_link = video_dir / "rgb"
            video_dir.mkdir(parents=True, exist_ok=True)
            if rgb_link.exists() or rgb_link.is_symlink():
                if rgb_link.is_symlink() and rgb_link.resolve() == frame_dir.resolve():
                    pass
                else:
                    raise FileExistsError(rgb_link)
            else:
                rgb_link.symlink_to(frame_dir, target_is_directory=True)
            annotations.append(
                {
                    "id": video_id,
                    "video": f"images/{video_id}",
                    "instructions": [str(item.get("instruction", ""))],
                    "actions": _read_gt_actions(item),
                    "source_sample_id": video_id,
                    "source_frame_dir": str(frame_dir),
                    "source_split": item.get("split"),
                    "source_episode_id": item.get("episode_id"),
                }
            )
            if args.max_episodes > 0 and len(annotations) >= args.max_episodes:
                break

    out_annotations = args.output_root / "annotations.json"
    out_annotations.write_text(json.dumps(annotations, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "source_render_root": str(args.render_root),
        "output_root": str(args.output_root),
        "split": args.split,
        "episodes": len(annotations),
        "image_mode": "symlink_only",
        "annotations": str(out_annotations),
    }
    (args.output_root / "adapter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
