#!/usr/bin/env python3
"""把统一 episode manifest 增量缓存为多 chunk Wan VAE latent。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import torch
from torchvision.transforms.functional import center_crop

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "Infinite-World"))

from infworld.vae import WanVAEModelWrapper


def preprocess(image, height: int, width: int) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = max(height / image.shape[0], width / image.shape[1])
    image = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return center_crop(
        torch.from_numpy(image).permute(2, 0, 1), [height, width]
    )


def read_images(paths: list[str], height: int, width: int) -> torch.Tensor:
    frames = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"无法读取图像：{path}")
        frames.append(preprocess(image, height, width))
    return torch.stack(frames, dim=1).float().div_(127.5).sub_(1.0)


def read_video(
    path: str, indices: list[int], height: int, width: int
) -> torch.Tensor:
    wanted = set(indices)
    frames = {}
    capture = cv2.VideoCapture(path)
    first = min(indices)
    # canonical full-episode cache 会按固定 chunk 多次读取同一长视频。
    # 从0开始扫描会令第N个 chunk 重复解码前N-1个 chunks，形成 O(N²)。
    # 先 seek 到当前窗口首帧；若后端不支持精确 seek，则回退到从0扫描。
    if first > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    if position > first or position < 0:
        capture.release()
        capture = cv2.VideoCapture(path)
        position = 0
    last = max(indices)
    while position <= last:
        ok, image = capture.read()
        if not ok:
            break
        if position in wanted:
            frames[position] = preprocess(image, height, width)
        position += 1
    capture.release()
    missing = [index for index in indices if index not in frames]
    if missing:
        raise ValueError(f"{path} 缺少帧：{missing[:8]}")
    return torch.stack([frames[index] for index in indices], dim=1).float().div_(127.5).sub_(1.0)


def maybe_link_re10k(item: dict, output: Path) -> bool:
    if item["dataset"] != "re10k":
        return False
    source = (
        Path("/sharedata/RealEstate10K/nav_register/latents")
        / f'{item["episode_id"]}.pt'
    )
    if not source.is_file():
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, output)
    except FileExistsError:
        pass
    except OSError:
        return False
    return output.is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("/sharedata/NAV/derived/manifests/episodes.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/sharedata/NAV/derived/latents"),
    )
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--min-chunks", type=int, default=2)
    parser.add_argument(
        "--datasets",
        default="",
        help="可选逗号分隔数据集过滤，例如 dl3dv,argoverse2",
    )
    parser.add_argument(
        "--max-cache-chunks", type=int, default=0,
        help="每条最多缓存多少 chunk；0 表示缓存完整 episode",
    )
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in args.manifest.open()
        if line.strip()
    ]
    requested = {
        name.strip() for name in args.datasets.split(",") if name.strip()
    }
    items = [
        item for index, item in enumerate(items)
        if item["num_chunks"] >= args.min_chunks
        and (not requested or item["dataset"] in requested)
        and index % args.num_shards == args.shard_index
    ]
    if args.max_episodes:
        items = items[:args.max_episodes]

    vae = None
    completed = errors = linked = 0
    for position, item in enumerate(items, 1):
        output = args.output / item["dataset"] / f'{item["sample_id"]}.pt'
        sidecar = output.with_suffix(".json")
        target_chunks = (
            item["num_chunks"] if args.max_cache_chunks == 0
            else min(item["num_chunks"], args.max_cache_chunks)
        )
        existing_payload = None
        existing_chunks = 0
        if output.is_file():
            if sidecar.is_file():
                existing_chunks = int(json.loads(sidecar.read_text())["cached_chunks"])
            else:
                # 旧缓存均为完整 episode。
                existing_chunks = item["num_chunks"]
            if existing_chunks >= target_chunks:
                completed += 1
                continue
            existing_payload = torch.load(
                output, map_location="cpu", weights_only=False
            )
        if maybe_link_re10k(item, output):
            linked += 1
            print(json.dumps({"linked": str(output)}), flush=True)
            continue
        if vae is None:
            vae = WanVAEModelWrapper(
                vae_pth="/sharedata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                dtype=torch.bfloat16,
                device=args.device,
            ).to(args.device)
        try:
            latents = (
                [existing_payload["chunks"][:, index] for index in range(existing_chunks)]
                if existing_payload is not None else []
            )
            chunk_frames = item["chunk_frames"]
            for chunk_index in range(existing_chunks, target_chunks):
                start = chunk_index * chunk_frames
                end = start + chunk_frames
                if item["source_type"] == "images":
                    video = read_images(
                        item["frame_paths"][start:end], args.height, args.width
                    )
                else:
                    video = read_video(
                        item["video_path"],
                        item["frame_indices"][start:end],
                        args.height,
                        args.width,
                    )
                with torch.inference_mode():
                    latent = vae.encode(video[None].to(args.device)).cpu()
                latents.append(latent)
                del video, latent
                torch.cuda.empty_cache()
            payload = {
                "sample_id": item["sample_id"],
                "dataset": item["dataset"],
                "chunks": torch.stack(latents, dim=1),
                "move": torch.tensor(item["move"], dtype=torch.long),
                "view": torch.tensor(item["view"], dtype=torch.long),
                "num_chunks": target_chunks,
                "source_num_chunks": item["num_chunks"],
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(output)
            sidecar_tmp = sidecar.with_suffix(".json.tmp")
            sidecar_tmp.write_text(json.dumps({
                "sample_id": item["sample_id"],
                "cached_chunks": target_chunks,
                "source_num_chunks": item["num_chunks"],
            }))
            sidecar_tmp.replace(sidecar)
            completed += 1
            print(
                json.dumps(
                    {
                        "cached": str(output),
                        "position": position,
                        "total": len(items),
                        "num_chunks": target_chunks,
                        "source_num_chunks": item["num_chunks"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            errors += 1
            print(
                json.dumps(
                    {"error": item["sample_id"], "message": repr(error)}
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "selected": len(items),
                "completed": completed,
                "linked": linked,
                "errors": errors,
                "output": str(args.output),
            }
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
