#!/usr/bin/env python3
"""Run open-loop GigaNav inference on real R2R latent/text windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giga_nav import GigaNavConfig, GigaNavDataConfig, GigaNavModel, GigaNavR2RBatchBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_keys = set(GigaNavConfig.__dataclass_fields__)
    data_keys = set(GigaNavDataConfig.__dataclass_fields__)
    cfg_values = {key: value for key, value in payload.get("model_config", {}).items() if key in model_keys}
    if "backbone_checkpoint" in cfg_values:
        cfg_values["backbone_checkpoint"] = Path(cfg_values["backbone_checkpoint"])
    if "patch_size" in cfg_values:
        cfg_values["patch_size"] = tuple(cfg_values["patch_size"])
    cfg = GigaNavConfig(**cfg_values)
    data_values = {key: value for key, value in payload.get("data_config", {}).items() if key in data_keys}
    # Checkpoints written before 2026-09-05 used the old +12-step target
    # semantics.  Preserve their open-loop reproduction instead of silently
    # evaluating them with the corrected reference-frame labels.
    if "action_label_alignment" not in data_values:
        data_values["action_label_alignment"] = "after_observation_chunk"
    for key in ("latent_manifest_dir", "rendered_manifest", "text_empty", "text_cache_root"):
        if key in data_values:
            data_values[key] = Path(data_values[key])
    data_cfg = GigaNavDataConfig(**data_values)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    data = GigaNavR2RBatchBuilder(data_cfg)
    model = GigaNavModel(cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device=device, dtype=dtype).eval()
    total = correct = 0
    rows = []
    with torch.inference_mode():
        for _ in range(args.batches):
            batch, meta = data.next_batch(device=device, dtype=dtype)
            output = model.forward_policy(obs_latent=batch["obs_latent"], text_embedding=batch["text_embedding"], text_mask=batch["text_mask"], state=batch["state"], action_noise=batch["action_noise"])
            pred, target = output["action_logits"].argmax(-1), batch["action_target"]
            mask = batch["action_loss_mask"].bool()
            total += int(mask.sum())
            correct += int(((pred == target) & mask).sum())
            rows.append({"sample_ids": meta.get("sample_ids"), "pred": pred.cpu().tolist(), "target": target.cpu().tolist()})
    result = {"batches": args.batches, "tokens": total, "accuracy": correct / max(total, 1), "rows": rows, "structure": model.structural_report()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({key: result[key] for key in ("batches", "tokens", "accuracy")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
