#!/usr/bin/env python3
"""Build a budgeted VLN episode manifest for rendered T4 micro latent prep.

目标是为 Stage3 VLN 渲染/latent 准备选出一个可复现子集。预算单位按当前
V1 正式 T_latent=4 micro chunk 计算：

  latent shape = [16, 4, 56, 112], dtype=fp16
  bytes_per_micro_chunk = 16 * 4 * 56 * 112 * 2

脚本只写新的 manifest，不复制 action JSON；输出 rows 继续引用 raw policy 中的
absolute ``episode_action_path``，可直接交给 ``render_v1_stage3_vln_obs.py``。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RAW_ROOT = Path("/sharedata/NAV/derived/v1/vln/raw_policy/20260810_030728")
DEFAULT_OUT_ROOT = Path("/sharedata/NAV/derived/v1/vln/raw_policy_budgeted")
BYTES_PER_T4_MICRO = 16 * 4 * 56 * 112 * 2


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def open_write(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt")
    return path.open("w")


def source_key(row: dict[str, Any]) -> str:
    return f"{row['dataset']}:{row['variant']}:{row['split']}"


def row_has_stop(row: dict[str, Any], *, assume_terminal_stop: bool, verify_action_json: bool) -> bool:
    if bool(row.get("has_stop", False)):
        return True
    if assume_terminal_stop and not verify_action_json:
        return True
    action_path = row.get("episode_action_path")
    if not action_path:
        return False
    try:
        payload = json.loads(Path(action_path).read_text())
        return any(int(x) == 0 for x in payload.get("gt_actions", []))
    except Exception:
        return False


def num_frames(
    row: dict[str, Any],
    *,
    post_stop_pad_frames: int,
    post_stop_min_frames: int,
    assume_terminal_stop: bool,
    verify_action_json: bool,
) -> int:
    frames = int(row.get("n_actions", 0)) + 1
    if row_has_stop(row, assume_terminal_stop=assume_terminal_stop, verify_action_json=verify_action_json):
        frames = max(
            frames + max(0, int(post_stop_pad_frames)),
            max(1, int(post_stop_min_frames)),
        )
    return frames


def num_micro_chunks(
    row: dict[str, Any],
    *,
    micro_frames: int,
    micro_stride: int,
    post_stop_pad_frames: int,
    post_stop_min_frames: int,
    assume_terminal_stop: bool,
    verify_action_json: bool,
) -> int:
    n = num_frames(
        row,
        post_stop_pad_frames=post_stop_pad_frames,
        post_stop_min_frames=post_stop_min_frames,
        assume_terminal_stop=assume_terminal_stop,
        verify_action_json=verify_action_json,
    )
    if n < micro_frames:
        return 0
    return (n - micro_frames) // micro_stride + 1


def parse_sources(payload: str) -> list[str]:
    return [item.strip() for item in payload.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", default=f"stage3_vln_budget_t4_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument(
        "--sources",
        default="rxr_ce:guide:train,rxr_ce:follower:train",
        help="逗号分隔，按顺序从这些 source 中选 episode。",
    )
    parser.add_argument(
        "--target-latent-gib",
        type=float,
        default=500.0,
        help="最终组合目标 latent GiB；如果设置 preexisting-latent-gib，则本 manifest 只补剩余预算。",
    )
    parser.add_argument("--preexisting-latent-gib", type=float, default=0.0)
    parser.add_argument("--micro-frames", type=int, default=13)
    parser.add_argument("--micro-stride", type=int, default=12)
    parser.add_argument("--min-micro-chunks", type=int, default=1)
    parser.add_argument(
        "--post-stop-pad-frames",
        type=int,
        default=12,
        help="与 render_v1_stage3_vln_obs.py 对齐：STOP 后额外重复 terminal observation 的帧数。",
    )
    parser.add_argument(
        "--post-stop-min-frames",
        type=int,
        default=97,
        help="与 render_v1_stage3_vln_obs.py 对齐：含 STOP episode 的最小渲染帧数。",
    )
    parser.add_argument(
        "--no-assume-terminal-stop",
        action="store_true",
        help="默认假设 raw VLN episode 由 build_v1_stage3_vln_raw.py 生成且含 terminal STOP；设置该项则不默认假设。",
    )
    parser.add_argument(
        "--verify-stop-from-action-json",
        action="store_true",
        help="逐个读取 episode_action_path 判断 STOP；更准确但很慢，通常无需开启。",
    )
    parser.add_argument(
        "--selection",
        choices=("ordered", "round_robin"),
        default="round_robin",
        help="ordered 按 sources 顺序填满；round_robin 在 source 间轮询，避免单一 source 过重。",
    )
    parser.add_argument("--compress", action="store_true", default=True)
    args = parser.parse_args()

    raw_manifest = args.raw_root / "manifests" / "stage3_vln_episodes.jsonl.gz"
    if not raw_manifest.exists():
        raise FileNotFoundError(raw_manifest)

    target_bytes = int(args.target_latent_gib * (1024**3))
    preexisting_bytes = int(args.preexisting_latent_gib * (1024**3))
    needed_bytes = max(0, target_bytes - preexisting_bytes)
    needed_chunks = math.ceil(needed_bytes / BYTES_PER_T4_MICRO)
    sources = parse_sources(args.sources)
    source_set = set(sources)

    rows_by_source: dict[str, list[dict[str, Any]]] = {key: [] for key in sources}
    all_source_stats: dict[str, dict[str, int]] = {}
    for row in iter_jsonl(raw_manifest):
        key = source_key(row)
        frames = num_frames(
            row,
            post_stop_pad_frames=args.post_stop_pad_frames,
            post_stop_min_frames=args.post_stop_min_frames,
            assume_terminal_stop=not args.no_assume_terminal_stop,
            verify_action_json=args.verify_stop_from_action_json,
        )
        chunks = num_micro_chunks(
            row,
            micro_frames=args.micro_frames,
            micro_stride=args.micro_stride,
            post_stop_pad_frames=args.post_stop_pad_frames,
            post_stop_min_frames=args.post_stop_min_frames,
            assume_terminal_stop=not args.no_assume_terminal_stop,
            verify_action_json=args.verify_stop_from_action_json,
        )
        stats = all_source_stats.setdefault(key, {"episodes": 0, "frames": 0, "micro_chunks": 0})
        stats["episodes"] += 1
        stats["frames"] += frames
        stats["micro_chunks"] += chunks
        if key not in source_set or chunks < args.min_micro_chunks:
            continue
        row = dict(row)
        row["v1_t4_micro_frames"] = args.micro_frames
        row["v1_t4_micro_stride"] = args.micro_stride
        row["v1_t4_num_micro_chunks"] = chunks
        row["v1_t4_estimated_render_frames"] = frames
        row["v1_t4_post_stop_pad_frames"] = args.post_stop_pad_frames
        row["v1_t4_post_stop_min_frames"] = args.post_stop_min_frames
        row["v1_t4_latent_bytes"] = chunks * BYTES_PER_T4_MICRO
        rows_by_source[key].append(row)

    selected: list[dict[str, Any]] = []
    selected_chunks = 0
    if args.selection == "ordered":
        for key in sources:
            for row in rows_by_source.get(key, []):
                if selected_chunks >= needed_chunks:
                    break
                selected.append(row)
                selected_chunks += int(row["v1_t4_num_micro_chunks"])
            if selected_chunks >= needed_chunks:
                break
    else:
        positions = {key: 0 for key in sources}
        while selected_chunks < needed_chunks:
            progressed = False
            for key in sources:
                pos = positions[key]
                rows = rows_by_source.get(key, [])
                if pos >= len(rows):
                    continue
                row = rows[pos]
                positions[key] = pos + 1
                selected.append(row)
                selected_chunks += int(row["v1_t4_num_micro_chunks"])
                progressed = True
                if selected_chunks >= needed_chunks:
                    break
            if not progressed:
                break

    run_root = args.out_root / args.run_name
    suffix = ".jsonl.gz" if args.compress else ".jsonl"
    out_manifest = run_root / "manifests" / f"stage3_vln_episodes{suffix}"
    summary_path = run_root / "manifests" / "budget_summary.json"

    per_source: dict[str, dict[str, int]] = {}
    with open_write(out_manifest) as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            key = source_key(row)
            stats = per_source.setdefault(key, {"episodes": 0, "frames": 0, "micro_chunks": 0})
            stats["episodes"] += 1
            stats["frames"] += int(row["v1_t4_estimated_render_frames"])
            stats["micro_chunks"] += int(row["v1_t4_num_micro_chunks"])

    selected_frames = sum(int(row["v1_t4_estimated_render_frames"]) for row in selected)
    selected_bytes = selected_chunks * BYTES_PER_T4_MICRO
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw_root": str(args.raw_root),
        "run_root": str(run_root),
        "manifest": str(out_manifest),
        "sources": sources,
        "selection": args.selection,
        "target_latent_gib_combined": args.target_latent_gib,
        "preexisting_latent_gib": args.preexisting_latent_gib,
        "needed_latent_gib": needed_bytes / 1024**3,
        "bytes_per_t4_micro_chunk": BYTES_PER_T4_MICRO,
        "micro_frames": args.micro_frames,
        "micro_stride": args.micro_stride,
        "post_stop_pad_frames": args.post_stop_pad_frames,
        "post_stop_min_frames": args.post_stop_min_frames,
        "assume_terminal_stop": not args.no_assume_terminal_stop,
        "verify_stop_from_action_json": bool(args.verify_stop_from_action_json),
        "target_needed_micro_chunks": needed_chunks,
        "selected_episodes": len(selected),
        "selected_frames": selected_frames,
        "selected_micro_chunks": selected_chunks,
        "selected_latent_gib": selected_bytes / 1024**3,
        "selected_latent_gb_decimal": selected_bytes / 1e9,
        "estimated_raw_png_gib_at_260461_bytes_per_frame": selected_frames * 260461 / 1024**3,
        "per_source": per_source,
        "all_source_stats": all_source_stats,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    latest = args.out_root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_root, target_is_directory=True)
    except OSError:
        pass
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
