#!/usr/bin/env python3
"""Measure GigaNav policy sensitivity to instruction replacement.

The observation, state, action-query slots and target are held fixed.  Each
batch is evaluated with (1) its matched R2R instruction, (2) a cyclically
shuffled instruction from another sample, and (3) the canonical empty-prompt
embedding used by training.  This is a full-checkpoint/full-backbone test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giga_nav import (  # noqa: E402
    GigaNavConfig,
    GigaNavDataConfig,
    GigaNavModel,
    GigaNavR2RBatchBuilder,
)
from nav.v1.data.r2r import ACTION_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _config_from_checkpoint(payload: dict[str, Any], batch_size: int) -> tuple[GigaNavConfig, GigaNavDataConfig]:
    model_keys = set(GigaNavConfig.__dataclass_fields__)
    data_keys = set(GigaNavDataConfig.__dataclass_fields__)
    model_values = {key: value for key, value in payload.get("model_config", {}).items() if key in model_keys}
    if "backbone_checkpoint" in model_values:
        model_values["backbone_checkpoint"] = Path(model_values["backbone_checkpoint"])
    if "patch_size" in model_values:
        model_values["patch_size"] = tuple(model_values["patch_size"])
    data_values = {key: value for key, value in payload.get("data_config", {}).items() if key in data_keys}
    for key in ("latent_manifest_dir", "rendered_manifest", "text_empty", "text_cache_root"):
        if key in data_values:
            data_values[key] = Path(data_values[key])
    data_values["batch_size"] = batch_size
    return GigaNavConfig(**model_values), GigaNavDataConfig(**data_values)


def _masked_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
    valid = mask.bool()
    return int(((pred == target) & valid).sum().item()), int(valid.sum().item())


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be >= 2 so shuffled instructions are from another sample")
    if args.samples < args.batch_size:
        raise ValueError("--samples must be >= --batch-size")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg, data_cfg = _config_from_checkpoint(payload, args.batch_size)
    data = GigaNavR2RBatchBuilder(data_cfg)
    model = GigaNavModel(model_cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device=device, dtype=dtype).eval()

    empty_y = data.base.empty_y.to(device=device, dtype=dtype)
    empty_mask = data.base.empty_mask.to(device=device, dtype=dtype)
    scenario_correct = {name: 0 for name in ("matched", "shuffled", "empty")}
    scenario_total = {name: 0 for name in scenario_correct}
    token_flip = {name: 0 for name in ("shuffled", "empty")}
    any_horizon_flip = {name: 0 for name in token_flip}
    first_action_flip = {name: 0 for name in token_flip}
    total_variation_sum = {name: 0.0 for name in token_flip}
    hidden_l2_sum = {name: 0.0 for name in token_flip}
    compared_tokens = 0
    compared_samples = 0
    rows: list[dict[str, Any]] = []
    batches = math.ceil(args.samples / args.batch_size)

    with torch.inference_mode():
        for _ in range(batches):
            batch, meta = data.next_batch(device=device, dtype=dtype)
            take = min(args.batch_size, args.samples - compared_samples)
            if take < 2:
                break
            common = {
                "obs_latent": batch["obs_latent"][:take],
                "state": batch["state"][:take],
                "action_noise": batch["action_noise"][:take],
            }
            matched_y = batch["text_embedding"][:take]
            matched_mask = batch["text_mask"][:take]
            variants = {
                "matched": (matched_y, matched_mask),
                "shuffled": (torch.roll(matched_y, shifts=1, dims=0), torch.roll(matched_mask, shifts=1, dims=0)),
                "empty": (
                    empty_y.expand(take, *empty_y.shape[1:]),
                    empty_mask.expand(take, *empty_mask.shape[1:]),
                ),
            }
            outputs: dict[str, dict[str, torch.Tensor]] = {}
            predictions: dict[str, torch.Tensor] = {}
            probabilities: dict[str, torch.Tensor] = {}
            target = batch["action_target"][:take]
            mask = batch["action_loss_mask"][:take]
            for name, (text, text_mask) in variants.items():
                outputs[name] = model.forward_policy(
                    **common,
                    text_embedding=text,
                    text_mask=text_mask,
                )
                logits = outputs[name]["action_logits"].float()
                predictions[name] = logits.argmax(dim=-1)
                probabilities[name] = logits.softmax(dim=-1)
                correct, total = _masked_accuracy(predictions[name], target, mask)
                scenario_correct[name] += correct
                scenario_total[name] += total

            base_pred = predictions["matched"]
            base_prob = probabilities["matched"]
            valid = mask.bool()
            valid_count = int(valid.sum().item())
            compared_tokens += valid_count
            compared_samples += take
            for name in ("shuffled", "empty"):
                changed = predictions[name] != base_pred
                token_flip[name] += int((changed & valid).sum().item())
                any_horizon_flip[name] += int((changed & valid).any(dim=1).sum().item())
                first_action_flip[name] += int(changed[:, 0].sum().item())
                tv = 0.5 * (probabilities[name] - base_prob).abs().sum(dim=-1)
                total_variation_sum[name] += float((tv * mask.float()).sum().item())
                hidden_delta = (outputs[name]["action_hidden"].float() - outputs["matched"]["action_hidden"].float()).norm(dim=-1)
                hidden_l2_sum[name] += float((hidden_delta * mask.float()).sum().item())

            ids = list(meta.get("sample_ids", []))[:take]
            shuffled_ids = [ids[(index - 1) % take] for index in range(take)]
            for index in range(take):
                rows.append(
                    {
                        "sample_id": ids[index],
                        "shuffled_instruction_from": shuffled_ids[index],
                        "target": target[index].cpu().tolist(),
                        "pred_matched": predictions["matched"][index].cpu().tolist(),
                        "pred_shuffled": predictions["shuffled"][index].cpu().tolist(),
                        "pred_empty": predictions["empty"][index].cpu().tolist(),
                    }
                )

    scenarios = {
        name: {
            "correct_tokens": scenario_correct[name],
            "valid_tokens": scenario_total[name],
            "token_accuracy": scenario_correct[name] / max(scenario_total[name], 1),
        }
        for name in scenario_correct
    }
    paired = {
        name: {
            "token_flip_rate_vs_matched": token_flip[name] / max(compared_tokens, 1),
            "any_horizon_flip_rate_vs_matched": any_horizon_flip[name] / max(compared_samples, 1),
            "first_action_flip_rate_vs_matched": first_action_flip[name] / max(compared_samples, 1),
            "mean_total_variation_vs_matched": total_variation_sum[name] / max(compared_tokens, 1),
            "mean_action_hidden_l2_vs_matched": hidden_l2_sum[name] / max(compared_tokens, 1),
        }
        for name in token_flip
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "device": str(device),
        "samples": compared_samples,
        "batch_size": args.batch_size,
        "protocol": "Hold obs/state/action slots fixed; replace only UMT5 instruction embedding and mask.",
        "scenarios": scenarios,
        "paired_change_vs_matched": paired,
        "action_names": {str(key): value for key, value in ACTION_NAMES.items()},
        "rows": rows,
        "structure": model.structural_report(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({"samples": compared_samples, "scenarios": scenarios, "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
