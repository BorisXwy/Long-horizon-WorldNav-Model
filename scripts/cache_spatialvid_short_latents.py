#!/usr/bin/env python3
"""独立于 action 标注缓存 SpatialVID 2–3 chunk VAE latent。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "Infinite-World"))

from cache_multidataset_latents import read_video
from infworld.vae import WanVAEModelWrapper
from nav.spatialvid_short_data import (
    iter_spatialvid_videos,
    select_short_window,
)


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
        default=Path("/sharedata/NAV/derived/latents/spatialvid"),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--max-chunks", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("num_shards 必须 >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index 必须位于 [0, num_shards)")

    args.output.mkdir(parents=True, exist_ok=True)
    vae = None
    counts = Counter()
    videos = (
        video
        for index, video in enumerate(iter_spatialvid_videos(args.root))
        if index % args.num_shards == args.shard_index
    )
    if args.max_episodes:
        videos = islice(videos, args.max_episodes)
    for position, video_path in enumerate(videos, 1):
        sample_id = f"spatialvid__{video_path.stem}"
        output = args.output / f"{sample_id}.pt"
        sidecar = output.with_suffix(".json")
        if output.is_file():
            counts["existing"] += 1
            continue
        try:
            frame_indices, window = select_short_window(
                video_path, max_chunks=args.max_chunks
            )
            if vae is None:
                vae = WanVAEModelWrapper(
                    vae_pth=(
                        "/sharedata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
                    ),
                    dtype=torch.bfloat16,
                    device=args.device,
                ).to(args.device)
            latents = []
            for chunk_index in range(window["cached_chunks"]):
                begin = chunk_index * window["chunk_frames"]
                end = begin + window["chunk_frames"]
                video = read_video(
                    str(video_path),
                    frame_indices[begin:end].tolist(),
                    args.height,
                    args.width,
                )
                with torch.inference_mode():
                    latent = vae.encode(
                        video[None].to(args.device)
                    ).cpu()
                latents.append(latent)
                del video, latent
                torch.cuda.empty_cache()
            payload = {
                "sample_id": sample_id,
                "dataset": "spatialvid",
                "chunks": torch.stack(latents, dim=1),
                "num_chunks": window["cached_chunks"],
                **window,
            }
            temporary = output.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(output)
            sidecar_tmp = sidecar.with_suffix(".json.tmp")
            sidecar_tmp.write_text(json.dumps({
                "sample_id": sample_id,
                "external_actions_required": True,
                **window,
            }) + "\n")
            sidecar_tmp.replace(sidecar)
            counts["completed"] += 1
            print(json.dumps({
                "position": position,
                "cached": str(output),
                **window,
            }), flush=True)
        except Exception as error:
            counts["errors"] += 1
            print(json.dumps({
                "position": position,
                "error": sample_id,
                "message": repr(error),
            }, ensure_ascii=False), flush=True)
    summary = {
        "output": str(args.output),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        **counts,
    }
    suffix = (
        "" if args.num_shards == 1
        else f".shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    )
    (args.output / f"_short_summary{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / f"_SHORT_COMPLETE{suffix}").touch()
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
