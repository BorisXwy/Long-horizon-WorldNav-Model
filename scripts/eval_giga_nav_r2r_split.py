#!/usr/bin/env python3
"""Evaluate a GigaNav checkpoint on a deterministic R2R-CE split subset.

This evaluator uses the complete Wan backbone, official UMT5 encoder, existing
per-frame Wan VAE latent cache, and GT action traces.  For legacy GigaNav
checkpoints trained with a +12 action-label offset, the input reference frame
is exactly 12 simulator steps before the first supervised action.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Infinite-World"))

from giga_nav import GigaNavConfig, GigaNavModel  # noqa: E402
from infworld.models.umt5 import T5EncoderModel  # noqa: E402
from nav.v1.data.r2r import ACTION_NAMES, STOP_ACTION  # noqa: E402


DEFAULT_LATENT_ROOT = Path(
    "/sharedata/NAV/derived/v1/vln/obs_latents_r2r_full/"
    "r2r_standard_20260817_full"
)
DEFAULT_ACTION_ROOT = Path(
    "/sharedata/NAV/derived/v1/vln/raw_policy/20260810_030728/"
    "actions/r2r_ce/standard"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val_seen", "val_unseen"), required=True)
    parser.add_argument("--max-episodes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--action-root", type=Path, default=DEFAULT_ACTION_ROOT)
    parser.add_argument("--instruction-sensitivity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(root: Path, split: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in sorted((root / "manifests").glob("*.jsonl")):
        with manifest.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("split") == split and Path(row["latent_path"]).is_file():
                    rows.append(row)
    rows.sort(key=lambda row: int(row["episode_id"]))
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"no latent rows for split={split} under {root}")
    return rows


def load_action_payload(root: Path, split: str, episode_id: str) -> dict[str, Any]:
    path = root / split / f"ep{episode_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def deterministic_target_start(sample_id: str, action_count: int, offset: int) -> int:
    """Choose one reproducible post-offset target position per episode."""

    if action_count <= offset:
        return max(0, action_count - 1)
    span = action_count - offset
    digest = hashlib.blake2b(sample_id.encode(), digest_size=8).digest()
    return offset + int.from_bytes(digest, "little") % span


def padded_action_window(actions: list[int], start: int, horizon: int) -> list[int]:
    return [int(actions[index]) if index < len(actions) else STOP_ACTION for index in range(start, start + horizon)]


def model_config(payload: dict[str, Any]) -> GigaNavConfig:
    keys = set(GigaNavConfig.__dataclass_fields__)
    values = {key: value for key, value in payload.get("model_config", {}).items() if key in keys}
    if "backbone_checkpoint" in values:
        values["backbone_checkpoint"] = Path(values["backbone_checkpoint"])
    if "patch_size" in values:
        values["patch_size"] = tuple(values["patch_size"])
    return GigaNavConfig(**values)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = model_config(checkpoint)
    label_offset = 0 if checkpoint.get("data_config", {}).get("action_label_alignment") == "reference_frame" else 12
    rows = load_rows(args.latent_root, args.split, args.max_episodes)

    examples: list[dict[str, Any]] = []
    instructions: list[str] = []
    for row in rows:
        action_payload = load_action_payload(args.action_root, args.split, str(row["episode_id"]))
        actions = [int(value) for value in action_payload["gt_actions"]]
        target_start = deterministic_target_start(str(row["sample_id"]), len(actions), label_offset)
        reference_index = max(0, target_start - label_offset)
        reference_index = min(reference_index, int(row["num_frames"]) - 1)
        examples.append(
            {
                "row": row,
                "reference_index": reference_index,
                "target_start": target_start,
                "target": padded_action_window(actions, target_start, cfg.action_horizon),
            }
        )
        instructions.append(str(action_payload.get("instruction") or row.get("instruction") or ""))

    # Encode the real instructions first, then release UMT5 before loading the
    # full policy model.  This keeps the peak on one 48-GiB device bounded.
    encoder = T5EncoderModel(
        model_max_length=cfg.text_length,
        dtype=dtype,
        device=str(device),
        checkpoint_path="/sharedata/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="/sharedata/Wan2.1-T2V-1.3B/google/umt5-xxl",
    )
    text_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for start in range(0, len(instructions), args.batch_size):
            encoded = encoder.encode(instructions[start : start + args.batch_size])
            text_batches.append((encoded["y"].cpu().to(torch.bfloat16), encoded["y_mask"].cpu()))
        empty_encoded = encoder.encode([""])
        empty_y = empty_encoded["y"].cpu().to(torch.bfloat16)
        empty_y_mask = empty_encoded["y_mask"].cpu()
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = GigaNavModel(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    confusion = torch.zeros(cfg.num_nav_classes, cfg.num_nav_classes, dtype=torch.long)
    prediction_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    scenario_names = ("matched", "shuffled", "empty") if args.instruction_sensitivity else ("matched",)
    scenario_correct = {name: 0 for name in scenario_names}
    token_flip = {name: 0 for name in scenario_names if name != "matched"}
    any_horizon_flip = {name: 0 for name in token_flip}
    first_action_flip = {name: 0 for name in token_flip}
    total_variation_sum = {name: 0.0 for name in token_flip}
    output_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(examples), args.batch_size)):
            current = examples[start : start + args.batch_size]
            obs = []
            target = []
            for example in current:
                latent_payload = torch.load(example["row"]["latent_path"], map_location="cpu", weights_only=False)
                reference = latent_payload["obs_latents"][example["reference_index"]]
                obs.append(torch.cat([reference, torch.zeros_like(reference).expand(-1, 3, -1, -1)], dim=1))
                target.append(example["target"])
            obs_tensor = torch.stack(obs).to(device=device, dtype=dtype)
            y, y_mask = text_batches[batch_index]
            target_tensor = torch.tensor(target, dtype=torch.long)
            text_variants = {"matched": (y, y_mask)}
            if args.instruction_sensitivity:
                text_variants.update(
                    {
                        "shuffled": (torch.roll(y, shifts=1, dims=0), torch.roll(y_mask, shifts=1, dims=0)),
                        "empty": (
                            empty_y.expand(len(current), *empty_y.shape[1:]),
                            empty_y_mask.expand(len(current), *empty_y_mask.shape[1:]),
                        ),
                    }
                )
            predictions: dict[str, torch.Tensor] = {}
            probabilities: dict[str, torch.Tensor] = {}
            for name, (variant_y, variant_mask) in text_variants.items():
                output = model.forward_policy(
                    obs_latent=obs_tensor,
                    text_embedding=variant_y.to(device=device, dtype=dtype),
                    text_mask=variant_mask.to(device=device),
                )
                logits = output["action_logits"].float().cpu()
                predictions[name] = logits.argmax(dim=-1)
                probabilities[name] = logits.softmax(dim=-1)
                scenario_correct[name] += int((predictions[name] == target_tensor).sum().item())
            pred = predictions["matched"]
            for name in token_flip:
                changed = predictions[name] != pred
                token_flip[name] += int(changed.sum().item())
                any_horizon_flip[name] += int(changed.any(dim=1).sum().item())
                first_action_flip[name] += int(changed[:, 0].sum().item())
                total_variation_sum[name] += float(
                    (0.5 * (probabilities[name] - probabilities["matched"]).abs().sum(dim=-1)).sum().item()
                )
            for p, t in zip(pred.reshape(-1), target_tensor.reshape(-1)):
                confusion[int(p), int(t)] += 1
                prediction_counts[int(p)] += 1
                target_counts[int(t)] += 1
            for index, example in enumerate(current):
                output_rows.append(
                    {
                        "sample_id": example["row"]["sample_id"],
                        "reference_index": example["reference_index"],
                        "target_start": example["target_start"],
                        "pred": pred[index].tolist(),
                        "target": target_tensor[index].tolist(),
                        "pred_shuffled": predictions.get("shuffled", pred)[index].tolist(),
                        "pred_empty": predictions.get("empty", pred)[index].tolist(),
                    }
                )

    correct = int(confusion.diagonal().sum().item())
    total = int(confusion.sum().item())
    per_class = {}
    for class_id, name in ACTION_NAMES.items():
        tp = int(confusion[class_id, class_id])
        predicted = int(confusion[class_id].sum())
        actual = int(confusion[:, class_id].sum())
        precision = tp / max(predicted, 1)
        recall = tp / max(actual, 1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "pred": predicted,
            "target": actual,
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": args.split,
        "episodes": len(examples),
        "tokens": total,
        "accuracy": correct / max(total, 1),
        "instruction_scenarios": {
            name: {
                "correct_tokens": scenario_correct[name],
                "tokens": total,
                "accuracy": scenario_correct[name] / max(total, 1),
            }
            for name in scenario_names
        },
        "instruction_change_vs_matched": {
            name: {
                "token_flip_rate": token_flip[name] / max(total, 1),
                "any_horizon_flip_rate": any_horizon_flip[name] / max(len(examples), 1),
                "first_action_flip_rate": first_action_flip[name] / max(len(examples), 1),
                "mean_total_variation": total_variation_sum[name] / max(total, 1),
            }
            for name in token_flip
        },
        "prediction_counts": {str(key): value for key, value in sorted(prediction_counts.items())},
        "target_counts": {str(key): value for key, value in sorted(target_counts.items())},
        "per_class": per_class,
        "confusion_pred_rows_target_cols": confusion.tolist(),
        "protocol": (
            "One deterministic window per episode; existing per-frame Wan latent; official UMT5; "
            f"checkpoint-derived label offset={label_offset}; H={cfg.action_horizon}; STOP padding after episode end."
        ),
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "split",
                    "episodes",
                    "tokens",
                    "accuracy",
                    "instruction_scenarios",
                    "instruction_change_vs_matched",
                    "per_class",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
