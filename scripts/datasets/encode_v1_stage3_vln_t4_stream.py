#!/usr/bin/env python3
"""Stream-encode rendered VLN episodes into V1 T_latent=4 micro latents.

This script is designed to run while ``render_v1_stage3_vln_obs.py`` is still
appending completed episodes.  For every completed rendered episode it:

1. reads rendered RGB PNG frames;
2. encodes independent 13-frame / stride-12 micro chunks with Wan VAE;
3. verifies the saved ``.pt`` file is readable;
4. optionally deletes only the PNG frames for that episode.

It deliberately keeps ``render_meta.json`` and writes ``latent_cleanup.json`` in
the frame directory, so the render manifest remains auditable while the large
RGB intermediate is reclaimed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import torch
from torchvision.transforms.functional import center_crop

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World")))

from infworld.vae import WanVAEModelWrapper  # noqa: E402


BYTES_PER_T4_MICRO = 16 * 4 * 56 * 112 * 2


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def iter_rendered_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    except (EOFError, gzip.BadGzipFile):
        # The renderer may still have the gzip stream open.  Treat this poll as
        # a partial read; the next pass will see a consistent suffix.
        return


def preprocess(path: Path, height: int, width: int) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = max(height / image.shape[0], width / image.shape[1])
    image = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return center_crop(torch.from_numpy(image).permute(2, 0, 1), [height, width])


def num_micro_chunks(n_frames: int, micro_frames: int, micro_stride: int) -> int:
    if n_frames < micro_frames:
        return 0
    return (n_frames - micro_frames) // micro_stride + 1


def episode_key(row: dict[str, Any]) -> str:
    return f"{row['dataset']}__{row['variant']}__{row['split']}__ep{row['episode_id']}"


def shard_matches(sample_id: str, shard_index: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    digest = hashlib.md5(sample_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:12], 16) % num_shards == shard_index


def output_path(output_root: Path, row: dict[str, Any]) -> Path:
    return output_root / str(row["dataset"]) / f"{episode_key(row)}.pt"


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def encoded_complete(path: Path, expected_micro: int) -> bool:
    sidecar = sidecar_path(path)
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text())
        if int(payload.get("cached_micro_chunks", -1)) != int(expected_micro):
            return False
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        return int(loaded.get("num_micro_chunks", -1)) == int(expected_micro)
    except Exception:
        return False


def read_micro_video(frame_dir: Path, micro_index: int, micro_frames: int, micro_stride: int, height: int, width: int) -> torch.Tensor:
    start = micro_index * micro_stride
    paths = [frame_dir / f"{idx:05d}.png" for idx in range(start, start + micro_frames)]
    frames = [preprocess(path, height, width) for path in paths]
    return torch.stack(frames, dim=1).float().div_(127.5).sub_(1.0)


def cleanup_pngs(frame_dir: Path, output: Path, expected_micro: int) -> dict[str, Any]:
    resolved = frame_dir.resolve()
    if "/rendered_obs/" not in str(resolved):
        raise ValueError(f"拒绝清理非 rendered_obs 路径：{resolved}")
    if not encoded_complete(output, expected_micro):
        raise ValueError(f"latent 未通过可读校验，拒绝清理：{output}")
    pngs = sorted(frame_dir.glob("*.png"))
    removed_bytes = 0
    removed = 0
    for path in pngs:
        try:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    marker = {
        "cleanup_time": now(),
        "mode": "png_deleted_after_t4_latent_verified",
        "latent_path": str(output),
        "expected_micro_chunks": expected_micro,
        "removed_png_files": removed,
        "removed_bytes": removed_bytes,
    }
    (frame_dir / "latent_cleanup.json").write_text(json.dumps(marker, indent=2, ensure_ascii=False) + "\n")
    return marker


def encode_one(
    *,
    vae: WanVAEModelWrapper,
    row: dict[str, Any],
    output_root: Path,
    device: str,
    height: int,
    width: int,
    micro_frames: int,
    micro_stride: int,
    delete_png_after_encode: bool,
) -> dict[str, Any]:
    frame_dir = Path(row["render_frame_dir"])
    n_frames = int(row["render_num_frames"])
    n_micro = num_micro_chunks(n_frames, micro_frames, micro_stride)
    output = output_path(output_root, row)
    if n_micro <= 0:
        return {"status": "skipped_short", "sample_id": episode_key(row), "frames": n_frames}
    if encoded_complete(output, n_micro):
        cleanup = None
        if delete_png_after_encode and any(frame_dir.glob("*.png")):
            cleanup = cleanup_pngs(frame_dir, output, n_micro)
        return {"status": "skipped_existing", "sample_id": episode_key(row), "num_micro_chunks": n_micro, "cleanup": cleanup}

    latents: list[torch.Tensor] = []
    for micro_index in range(n_micro):
        video = read_micro_video(frame_dir, micro_index, micro_frames, micro_stride, height, width)
        with torch.inference_mode():
            latent = vae.encode(video[None].to(device)).cpu()
        if tuple(latent.shape[2:])[:1] != (4,):
            raise ValueError(f"unexpected latent shape={tuple(latent.shape)} for {episode_key(row)}")
        latents.append(latent)
        del video, latent
        torch.cuda.empty_cache()

    payload = {
        "mode": "stage3_vln_t4_micro_latents",
        "sample_id": episode_key(row),
        "dataset": row["dataset"],
        "variant": row["variant"],
        "split": row["split"],
        "episode_id": str(row["episode_id"]),
        "trajectory_id": row.get("trajectory_id"),
        "scene_id": row.get("scene_id"),
        "instruction": row.get("instruction", ""),
        "micro_latents": torch.stack(latents, dim=1).to(torch.float16),
        "micro_frames": micro_frames,
        "micro_stride": micro_stride,
        "latent_t": 4,
        "num_micro_chunks": n_micro,
        "covered_frames": micro_frames + (n_micro - 1) * micro_stride,
        "source_num_frames": n_frames,
        "render_frame_dir": str(frame_dir),
        "render_run_root": row.get("render_run_root"),
        "action_path": row.get("episode_action_path"),
        "metadata": row,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(output)
    sidecar_path(output).write_text(
        json.dumps(
            {
                "sample_id": episode_key(row),
                "dataset": row["dataset"],
                "variant": row["variant"],
                "split": row["split"],
                "episode_id": str(row["episode_id"]),
                "cached_micro_chunks": n_micro,
                "micro_frames": micro_frames,
                "micro_stride": micro_stride,
                "latent_t": 4,
                "source_num_frames": n_frames,
                "latent_path": str(output),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    if not encoded_complete(output, n_micro):
        raise RuntimeError(f"保存后校验失败：{output}")

    cleanup = cleanup_pngs(frame_dir, output, n_micro) if delete_png_after_encode else None
    return {
        "status": "encoded",
        "sample_id": episode_key(row),
        "latent_path": str(output),
        "num_micro_chunks": n_micro,
        "cleanup": cleanup,
    }


def render_process_alive(render_run_name: str) -> bool:
    if not render_run_name:
        return True
    try:
        import subprocess

        result = subprocess.run(
            ["pgrep", "-af", "render_v1_stage3_vln_obs.py"],
            check=False,
            text=True,
            capture_output=True,
        )
        return render_run_name in result.stdout
    except Exception:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--micro-frames", type=int, default=13)
    parser.add_argument("--micro-stride", type=int, default=12)
    parser.add_argument("--vae-path", type=Path, default=Path("/sharedata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    parser.add_argument("--delete-png-after-encode", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--idle-exit-polls", type=int, default=0, help="0 means never exit while idle.")
    parser.add_argument("--render-run-name", default="", help="If set, used to decide whether renderer is still alive.")
    parser.add_argument("--max-new-episodes", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of stream encoder shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current stream encoder shard index.")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    manifest = args.render_root / "episodes" / "rendered_episodes.jsonl.gz"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_name = (
        "encoded_episodes.jsonl"
        if args.num_shards == 1
        else f"encoded_episodes_shard-{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    )
    summary_name = (
        "summary.json"
        if args.num_shards == 1
        else f"summary_shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    )
    run_manifest = args.output_root / "manifests" / manifest_name
    run_manifest.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / summary_name

    vae = WanVAEModelWrapper(
        vae_pth=str(args.vae_path),
        dtype=torch.bfloat16,
        device=args.device,
    ).to(args.device)

    completed = skipped_existing = skipped_short = errors = cleaned_pngs = cleaned_bytes = encoded_chunks = 0
    seen_encoded: set[str] = set()
    if run_manifest.exists():
        with run_manifest.open() as handle:
            for line in handle:
                if line.strip():
                    try:
                        seen_encoded.add(json.loads(line)["sample_id"])
                    except Exception:
                        pass

    idle_polls = 0
    while True:
        made_progress = False
        new_this_run = 0
        rows = list(iter_rendered_rows(manifest))
        for row in rows:
            if row.get("render_status") != "complete":
                continue
            sid = episode_key(row)
            if not shard_matches(sid, args.shard_index, args.num_shards):
                continue
            if sid in seen_encoded:
                # Still opportunistically clean PNGs for already encoded files.
                n_micro = num_micro_chunks(int(row["render_num_frames"]), args.micro_frames, args.micro_stride)
                out = output_path(args.output_root, row)
                if args.delete_png_after_encode and n_micro > 0 and encoded_complete(out, n_micro) and any(Path(row["render_frame_dir"]).glob("*.png")):
                    marker = cleanup_pngs(Path(row["render_frame_dir"]), out, n_micro)
                    cleaned_pngs += int(marker["removed_png_files"])
                    cleaned_bytes += int(marker["removed_bytes"])
                    made_progress = True
                continue
            try:
                result = encode_one(
                    vae=vae,
                    row=row,
                    output_root=args.output_root,
                    device=args.device,
                    height=args.height,
                    width=args.width,
                    micro_frames=args.micro_frames,
                    micro_stride=args.micro_stride,
                    delete_png_after_encode=args.delete_png_after_encode,
                )
                status = result["status"]
                if status == "encoded":
                    completed += 1
                    encoded_chunks += int(result["num_micro_chunks"])
                    cleanup = result.get("cleanup") or {}
                    cleaned_pngs += int(cleanup.get("removed_png_files", 0))
                    cleaned_bytes += int(cleanup.get("removed_bytes", 0))
                    with run_manifest.open("a") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    seen_encoded.add(sid)
                    new_this_run += 1
                    made_progress = True
                elif status == "skipped_existing":
                    skipped_existing += 1
                    seen_encoded.add(sid)
                    made_progress = True
                elif status == "skipped_short":
                    skipped_short += 1
                    seen_encoded.add(sid)
                print(json.dumps({"event": "stream_encode", "time": now(), **result}, ensure_ascii=False), flush=True)
            except Exception as exc:
                errors += 1
                print(json.dumps({"event": "stream_encode_error", "time": now(), "sample_id": sid, "error": repr(exc)}, ensure_ascii=False), flush=True)
            finally:
                torch.cuda.empty_cache()
            if args.max_new_episodes and new_this_run >= args.max_new_episodes:
                break

        summary = {
            "time": now(),
            "render_root": str(args.render_root),
            "output_root": str(args.output_root),
            "manifest": str(manifest),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "seen_rendered_rows": len(rows),
            "completed_encoded_this_process": completed,
            "skipped_existing_this_process": skipped_existing,
            "skipped_short_this_process": skipped_short,
            "errors_this_process": errors,
            "encoded_micro_chunks_this_process": encoded_chunks,
            "encoded_latent_gib_this_process": encoded_chunks * BYTES_PER_T4_MICRO / 1024**3,
            "cleaned_png_files_this_process": cleaned_pngs,
            "cleaned_gib_this_process": cleaned_bytes / 1024**3,
            "delete_png_after_encode": bool(args.delete_png_after_encode),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

        if args.max_new_episodes and completed >= args.max_new_episodes:
            break
        if made_progress:
            idle_polls = 0
        else:
            idle_polls += 1
        if args.idle_exit_polls and idle_polls >= args.idle_exit_polls and not render_process_alive(args.render_run_name):
            break
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
