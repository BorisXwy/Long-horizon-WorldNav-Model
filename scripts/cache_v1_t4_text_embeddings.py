#!/usr/bin/env python3
"""Cache per-sample UMT5 text conditions for V1 T4 Stage One.

当前用途：
- SpatialVID: 从官方 caption.json / instructions.json 构造文本条件；
- 其它数据集: 暂不生成 sidecar，训练脚本自动 fallback 到 empty UMT5。

该脚本只写 `/sharedata/NAV/derived/` 下的派生产物，不修改原始数据和 latent。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "Infinite-World"))

from infworld.models.umt5 import T5EncoderModel


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def annotation_paths(item: dict[str, Any]) -> tuple[Path | None, Path | None]:
    if item.get("dataset") != "spatialvid":
        return None, None
    video_path = Path(item["video_path"])
    try:
        group = video_path.parent.name
        stem = video_path.stem
    except Exception:
        return None, None
    base = Path("/sharedata/datasets/SpatialVID/raw/huggingface/SpatialVID/annotations")
    root = base / group / stem
    return root / "caption.json", root / "instructions.json"


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(normalize_text(x) for x in value if normalize_text(x))
    if isinstance(value, dict):
        return " ".join(f"{k}: {normalize_text(v)}" for k, v in value.items() if normalize_text(v))
    return " ".join(str(value).split())


def spatialvid_prompt(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    caption_path, instruction_path = annotation_paths(item)
    caption = read_json(caption_path)
    instruction = read_json(instruction_path)
    parts = []
    for key in ["SceneSummary", "SceneDescription", "CameraMotion", "ShotImmersion"]:
        value = normalize_text(caption.get(key))
        if value:
            label = {
                "SceneSummary": "Scene summary",
                "SceneDescription": "Scene description",
                "CameraMotion": "Camera motion",
                "ShotImmersion": "Shot immersion",
            }[key]
            parts.append(f"{label}: {value}")
    instruction_text = normalize_text(instruction)
    if instruction_text:
        parts.append(f"Camera instruction: {instruction_text}")
    text = " ".join(parts).strip()
    metadata = {
        "caption_path": str(caption_path) if caption_path else None,
        "instruction_path": str(instruction_path) if instruction_path else None,
        "has_caption": bool(caption),
        "has_instruction": bool(instruction),
    }
    return text, metadata


def build_items(manifest: Path, latent_root: Path, datasets: set[str]) -> list[dict[str, Any]]:
    existing = {
        path.stem
        for dataset in datasets
        for path in (latent_root / dataset).glob("*.pt")
        if path.is_file()
    }
    items = []
    seen = set()
    for item in iter_jsonl(manifest):
        dataset = item.get("dataset")
        sample_id = item.get("sample_id")
        if dataset not in datasets or sample_id not in existing or sample_id in seen:
            continue
        seen.add(sample_id)
        items.append(item)
    return items


def valid_cached_text(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return (
            "y" in payload
            and "y_mask" in payload
            and tuple(payload["y"].shape[1:]) == (1, 512, 4096)
            and tuple(payload["y_mask"].shape[1:]) == (512,)
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes_spatial20.jsonl"),
    )
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=Path("/sharedata/NAV/derived/v1/t4_micro_latents_spatial20"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/sharedata/NAV/derived/v1/text_embeddings/t4_micro"),
    )
    parser.add_argument("--datasets", default="spatialvid")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    items = build_items(args.manifest, args.latent_root, datasets)
    if args.max_items:
        items = items[: args.max_items]
    args.output_root.mkdir(parents=True, exist_ok=True)

    encoder = T5EncoderModel(
        model_max_length=512,
        dtype=torch.bfloat16,
        device=args.device,
        checkpoint_path="/sharedata/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="/sharedata/Wan2.1-T2V-1.3B/google/umt5-xxl",
    )

    completed = skipped = empty_text = 0
    texts: list[str] = []
    batch_items: list[dict[str, Any]] = []
    batch_meta: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal completed, skipped, empty_text, texts, batch_items, batch_meta
        if not texts:
            return
        with torch.inference_mode():
            encoded = encoder.encode(texts)
        y = encoded["y"].cpu().to(torch.bfloat16)
        y_mask = encoded["y_mask"].cpu()
        for index, item in enumerate(batch_items):
            dataset = item["dataset"]
            sample_id = item["sample_id"]
            output = args.output_root / dataset / f"{sample_id}.pt"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".pt.tmp")
            torch.save(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "episode_id": item.get("episode_id"),
                    "text": texts[index],
                    "text_source": batch_meta[index],
                    "y": y[index : index + 1],
                    "y_mask": y_mask[index : index + 1],
                },
                temporary,
            )
            temporary.replace(output)
            completed += 1
        print(json.dumps({
            "event": "cached_batch",
            "completed": completed,
            "total": len(items),
            "last": batch_items[-1]["sample_id"],
        }, ensure_ascii=False), flush=True)
        texts, batch_items, batch_meta = [], [], []

    for item in items:
        dataset = item["dataset"]
        sample_id = item["sample_id"]
        output = args.output_root / dataset / f"{sample_id}.pt"
        if valid_cached_text(output):
            skipped += 1
            continue
        if output.exists():
            output.unlink()
        if dataset == "spatialvid":
            text, meta = spatialvid_prompt(item)
        else:
            text, meta = "", {"reason": "dataset_without_text_recipe"}
        if not text:
            empty_text += 1
            continue
        texts.append(text)
        batch_items.append(item)
        batch_meta.append(meta)
        if len(texts) >= args.batch_size:
            flush()
    flush()

    summary = {
        "manifest": str(args.manifest),
        "latent_root": str(args.latent_root),
        "output_root": str(args.output_root),
        "datasets": sorted(datasets),
        "selected_items": len(items),
        "completed": completed,
        "skipped": skipped,
        "empty_text": empty_text,
    }
    summary_path = args.output_root / "_summary_latest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
