#!/usr/bin/env python3
"""用 Wan2.1 VAE 缓存少量 RE10K 流式训练样本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from torchvision.transforms.functional import center_crop

ROOT = Path(__file__).resolve().parents[1]
INFINITE_ROOT = ROOT.parent / "Infinite-World"
sys.path.insert(0, str(INFINITE_ROOT))

from infworld.vae import WanVAEModelWrapper


def read_chunk(paths: list[str], height: int, width: int) -> torch.Tensor:
    frames = []
    for path in paths:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        scale = max(height / image.shape[0], width / image.shape[1])
        resized = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        frame = torch.from_numpy(resized).permute(2, 0, 1)
        frames.append(center_crop(frame, [height, width]))
    video = torch.stack(frames, dim=1).float()
    return video.div_(127.5).sub_(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("/sharedata/RealEstate10K/nav_register/manifests/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/sharedata/RealEstate10K/nav_register/latents"))
    parser.add_argument(
        "--max-samples", type=int, default=0, help="0 表示处理全部"
    )
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    vae = WanVAEModelWrapper(
        vae_pth="/sharedata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        dtype=torch.bfloat16,
        device=args.device,
    ).to(args.device)
    with args.manifest.open() as stream, torch.inference_mode():
        for index, line in enumerate(stream):
            if args.max_samples and index >= args.max_samples:
                break
            if index % args.num_shards != args.shard_index:
                continue
            item = json.loads(line)
            output = args.output / f'{item["sample_id"]}.pt'
            if args.skip_existing and output.exists():
                print(json.dumps({"skipped": str(output)}))
                continue
            latents = []
            for start in (0, 81):
                video = read_chunk(
                    item["frame_paths"][start : start + 81], args.height, args.width
                ).unsqueeze(0).to(args.device)
                latents.append(vae.encode(video).cpu())
                del video
                torch.cuda.empty_cache()
            torch.save(
                {
                    "sample_id": item["sample_id"],
                    "chunks": torch.stack(latents, dim=1),
                    "move": torch.tensor(item["move"]),
                    "view": torch.tensor(item["view"]),
                    "frame_paths": item["frame_paths"],
                },
                output,
            )
            print(json.dumps({"cached": str(output), "shape": list(torch.stack(latents).shape)}))


if __name__ == "__main__":
    main()
