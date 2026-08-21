#!/usr/bin/env python3
"""Cache UMT5 instruction embeddings for V1 Stage3 VLN T4 latents.

The Stage3 latent manifest stores prepared VLN T4 micro-latents.  Each latent
sidecar contains ``dataset/variant/split/episode_id``; the natural language
instruction is read from the corresponding raw-policy action JSON and encoded
with Wan2.1's UMT5 text encoder.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "Infinite-World"))

from infworld.models.umt5 import T5EncoderModel  # noqa: E402


DEFAULT_MANIFEST = Path(
    "/sharedata/NAV/derived/v1/vln/t4_micro_latents_500g/rxr_budget500_stream_20260821_0000/"
    "manifests/encoded_episodes.jsonl"
)
DEFAULT_RAW_POLICY_ROOT = Path("/sharedata/NAV/derived/v1/vln/raw_policy/20260810_030728")
DEFAULT_OUTPUT_ROOT = Path("/sharedata/NAV/derived/v1/vln/text_embeddings/t4_micro")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def iter_manifest(path: Path):
    with open_text(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


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


def sidecar_metadata(latent_path: Path) -> dict[str, Any]:
    sidecar = latent_path.with_suffix(".json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text())
        except Exception:
            return {}
    return {}


def action_json_path(raw_policy_root: Path, meta: dict[str, Any]) -> Path | None:
    dataset = meta.get("dataset")
    variant = meta.get("variant")
    split = meta.get("split")
    episode_id = meta.get("episode_id")
    if not all([dataset, variant, split, episode_id]):
        return None
    return raw_policy_root / "actions" / str(dataset) / str(variant) / str(split) / f"ep{episode_id}.json"


def instruction_from_action(path: Path | None) -> tuple[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return "", {"action_path": str(path) if path is not None else None, "reason": "missing_action_json"}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return "", {"action_path": str(path), "reason": f"bad_action_json:{type(exc).__name__}"}
    text = normalize_text(payload.get("instruction"))
    return text, {
        "action_path": str(path),
        "has_instruction": bool(text),
        "dataset": payload.get("dataset"),
        "variant": payload.get("variant"),
        "split": payload.get("split"),
        "episode_id": payload.get("episode_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-policy-root", type=Path, default=DEFAULT_RAW_POLICY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for row in iter_manifest(args.manifest):
        latent_path = Path(row.get("latent_path", ""))
        if not latent_path.is_file():
            continue
        meta = sidecar_metadata(latent_path)
        if not meta:
            continue
        sample_id = meta.get("sample_id") or row.get("sample_id") or latent_path.stem
        output = args.output_root / str(meta.get("dataset", "unknown")) / f"{sample_id}.pt"
        rows.append({"manifest": row, "meta": meta, "sample_id": sample_id, "latent_path": latent_path, "output": output})
    if args.max_items > 0:
        rows = rows[: args.max_items]
    args.output_root.mkdir(parents=True, exist_ok=True)

    encoder = T5EncoderModel(
        model_max_length=512,
        dtype=torch.bfloat16,
        device=args.device,
        checkpoint_path="/sharedata/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="/sharedata/Wan2.1-T2V-1.3B/google/umt5-xxl",
    )

    completed = skipped = empty = missing_action = 0
    cached_manifest_rows: list[dict[str, Any]] = []
    texts: list[str] = []
    batch_rows: list[dict[str, Any]] = []
    batch_meta: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal completed, texts, batch_rows, batch_meta
        if not texts:
            return
        with torch.inference_mode():
            encoded = encoder.encode(texts)
        y = encoded["y"].cpu().to(torch.bfloat16)
        y_mask = encoded["y_mask"].cpu()
        for index, item in enumerate(batch_rows):
            output = item["output"]
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".pt.tmp")
            torch.save(
                {
                    "sample_id": item["sample_id"],
                    "dataset": item["meta"].get("dataset"),
                    "variant": item["meta"].get("variant"),
                    "split": item["meta"].get("split"),
                    "episode_id": item["meta"].get("episode_id"),
                    "text": texts[index],
                    "text_source": batch_meta[index],
                    "y": y[index : index + 1],
                    "y_mask": y_mask[index : index + 1],
                },
                temporary,
            )
            temporary.replace(output)
            cached_manifest_rows.append(item["manifest"])
            completed += 1
        print(
            json.dumps(
                {
                    "event": "cached_batch",
                    "completed": completed,
                    "total": len(rows),
                    "last": batch_rows[-1]["sample_id"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        texts, batch_rows, batch_meta = [], [], []

    for item in rows:
        output = item["output"]
        if valid_cached_text(output):
            cached_manifest_rows.append(item["manifest"])
            skipped += 1
            continue
        action_path = action_json_path(args.raw_policy_root, item["meta"])
        text, source = instruction_from_action(action_path)
        if not text:
            empty += 1
            if source.get("reason") == "missing_action_json":
                missing_action += 1
            continue
        texts.append(text)
        batch_rows.append(item)
        batch_meta.append(source)
        if len(texts) >= args.batch_size:
            flush()
    flush()

    summary = {
        "manifest": str(args.manifest),
        "raw_policy_root": str(args.raw_policy_root),
        "output_root": str(args.output_root),
        "selected_items": len(rows),
        "completed": completed,
        "skipped": skipped,
        "empty_instruction": empty,
        "missing_action_json": missing_action,
    }
    cached_manifest = args.output_root / "_cached_manifest_latest.jsonl"
    with cached_manifest.open("w") as handle:
        for row in cached_manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["cached_manifest"] = str(cached_manifest)
    (args.output_root / "_summary_latest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
