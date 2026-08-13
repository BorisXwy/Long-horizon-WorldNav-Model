#!/usr/bin/env python3
"""校验统一清单中的每条 episode 都已有可读取的 latent。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/sharedata/NAV/derived/manifests/episodes.jsonl"),
    )
    parser.add_argument(
        "--latents",
        type=Path,
        default=Path("/sharedata/NAV/derived/latents"),
    )
    parser.add_argument("--deep", action="store_true")
    parser.add_argument(
        "--required-max-chunks", type=int, default=0,
        help="0 要求完整缓存，否则要求 min(原长度, 此值)",
    )
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in args.manifest.open(encoding="utf-8")
        if line.strip()
    ]
    missing: list[str] = []
    invalid: list[str] = []
    for item in items:
        path = args.latents / item["dataset"] / f'{item["sample_id"]}.pt'
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
            continue
        sidecar = path.with_suffix(".json")
        cached_chunks = (
            int(json.loads(sidecar.read_text())["cached_chunks"])
            if sidecar.is_file() else item["num_chunks"]
        )
        required = (
            item["num_chunks"] if args.required_max_chunks == 0
            else min(item["num_chunks"], args.required_max_chunks)
        )
        if cached_chunks < required:
            missing.append(f"{path} ({cached_chunks}/{required} chunks)")
            continue
        if args.deep:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                chunks = payload["chunks"]
                if chunks.shape[1] < item["num_chunks"]:
                    raise ValueError(
                        f"chunks={chunks.shape[1]} < {item['num_chunks']}"
                    )
            except Exception as error:
                invalid.append(f"{path}: {error!r}")

    result = {
        "manifest_episodes": len(items),
        "ready": len(items) - len(missing) - len(invalid),
        "missing": len(missing),
        "invalid": len(invalid),
        "missing_examples": missing[:10],
        "invalid_examples": invalid[:10],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
