#!/usr/bin/env python3
"""Compare H1 and H4 Stage3 checkpoints on paired natural full-prefix R2R windows.

Both checkpoints run through the complete canonical Register + shared Wan
policy forward.  The H1 checkpoint is scored on the first future primitive;
the H4 checkpoint is scored on all four primitives and separately on slot 0,
which makes the direct comparison paired and unambiguous.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
INF_WORLD_ROOT = Path(os.environ.get("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World")))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(INF_WORLD_ROOT))

from nav.v1.data import (  # noqa: E402
    ACTION_NAMES,
    FullHistoryNaturalActionChunkR2RBatchBuilder,
    build_data_loader,
    r2r_data_config_from_dict,
)
from nav.v1.model import SingleActionStage3Model, build_model  # noqa: E402
from train_v1_stage2_final_cotrain import install_non_reentrant_wan_checkpoint, torch_dtype  # noqa: E402


CLASS_NAMES = [ACTION_NAMES[index] for index in sorted(ACTION_NAMES)]


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def adjacent_run_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint.parent.parent / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint-adjacent config: {path}")
    payload = json.loads(path.read_text())
    if "r2r_data" not in payload:
        raise RuntimeError(f"config has no r2r_data: {path}")
    return payload


def build_paired_loader(
    config_payload: dict[str, Any],
    *,
    batch_size: int,
    seed: int,
) -> FullHistoryNaturalActionChunkR2RBatchBuilder:
    cfg = r2r_data_config_from_dict(config_payload)
    cfg.action_horizon = 4
    cfg.batch_size = batch_size
    cfg.action_oversample_mode = "none"
    cfg.seed = seed
    builder = build_data_loader(
        "stage3_r2r_full_history_natural_action_chunk",
        config=cfg,
        history_action_horizon=10,
        model_action_horizon=10,
    )
    if not isinstance(builder, FullHistoryNaturalActionChunkR2RBatchBuilder):
        raise TypeError(type(builder))
    return builder


def load_model(
    checkpoint: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SingleActionStage3Model, dict[str, Any]]:
    assembly = build_model(
        "stage3_single_action_checkpoint",
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        training=False,
        require_policy_head=True,
    )
    model = assembly.model
    if not isinstance(model, SingleActionStage3Model):
        raise TypeError(type(model))
    audit = dict(assembly.audit)
    del assembly
    gc.collect()
    model.eval()
    return model, audit


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = diagonal / np.maximum(support, 1)
    precision = diagonal / np.maximum(predicted, 1)
    total = int(support.sum())
    return {
        "count": total,
        "accuracy": safe_div(int(diagonal.sum()), total),
        "macro_recall": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "majority_baseline_accuracy": safe_div(int(support.max()), total),
        "gt_count": {CLASS_NAMES[index]: int(support[index]) for index in range(4)},
        "pred_count": {CLASS_NAMES[index]: int(predicted[index]) for index in range(4)},
        "per_class": {
            CLASS_NAMES[index]: {
                "support": int(support[index]),
                "recall": float(recall[index]),
                "precision": float(precision[index]),
            }
            for index in range(4)
        },
        "confusion": confusion.tolist(),
    }


@torch.no_grad()
def evaluate_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    config_payload: dict[str, Any],
    num_windows: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    save_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[str, int]], torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, audit = load_model(checkpoint, device=device, dtype=dtype)
    policy_horizon = int(model.cfg.policy_queries)
    backbone_action_tokens = int(model.cfg.backbone_action_tokens)
    if policy_horizon not in {1, 4}:
        raise RuntimeError(f"comparison only supports H1/H4 checkpoints, got H={policy_horizon}")
    builder = build_paired_loader(config_payload, batch_size=batch_size, seed=seed)
    confusion_by_horizon = np.zeros((policy_horizon, 4, 4), dtype=np.int64)
    all_confusion = np.zeros((4, 4), dtype=np.int64)
    ce_sum = 0.0
    action_count = 0
    exact_sequences = 0
    sample_count = 0
    examples: list[dict[str, Any]] = []
    sample_keys: list[tuple[str, int]] = []
    predictions: list[torch.Tensor] = []
    by_history: dict[int, Counter[str]] = defaultdict(Counter)
    started = time.perf_counter()

    for batch_index in range(num_windows // batch_size):
        batch, meta = builder.next_batch(device=device, dtype=dtype)
        target = batch["action_class"][:, :policy_horizon].long()
        current_batch = target.shape[0]
        batch["action_class"] = target
        batch["action_combo"] = batch["action_combo"][:, :policy_horizon]
        batch["action_loss_mask"] = batch["action_loss_mask"][:, :policy_horizon]
        batch["action_target"] = batch["action_target"][:, :policy_horizon]
        batch["a_noise"] = torch.zeros(
            current_batch,
            backbone_action_tokens,
            6,
            device=device,
            dtype=dtype,
        )
        batch["a_cur_combo"] = torch.zeros(
            current_batch,
            backbone_action_tokens,
            device=device,
            dtype=torch.long,
        )
        batch["action_timestep"].zero_()
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model.forward_policy(batch)
        logits = out["action_logits"].float()
        pred = logits.argmax(dim=-1)
        ce_sum += float(
            F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1), reduction="sum").item()
        )
        action_count += int(target.numel())
        exact_sequences += int((pred == target).all(dim=1).sum().item())
        sample_count += current_batch
        predictions.append(pred.detach().cpu())

        history_lengths = [int(value) for value in meta["history_micro"]]
        for row_index in range(current_batch):
            key = (str(meta["sample_ids"][row_index]), int(meta["label_start_actions"][row_index]))
            sample_keys.append(key)
            row_target = target[row_index].detach().cpu()
            row_pred = pred[row_index].detach().cpu()
            history = history_lengths[row_index]
            by_history[history]["count"] += policy_horizon
            by_history[history]["correct"] += int((row_pred == row_target).sum().item())
            for horizon in range(policy_horizon):
                gt_id = int(row_target[horizon])
                pred_id = int(row_pred[horizon])
                confusion_by_horizon[horizon, gt_id, pred_id] += 1
                all_confusion[gt_id, pred_id] += 1
            if len(examples) < save_examples:
                examples.append(
                    {
                        "sample_id": key[0],
                        "label_start_action": key[1],
                        "history_micro": history,
                        "gt": [ACTION_NAMES[int(value)] for value in row_target.tolist()],
                        "pred": [ACTION_NAMES[int(value)] for value in row_pred.tolist()],
                    }
                )
        if (batch_index + 1) % 16 == 0:
            first = confusion_metrics(confusion_by_horizon[0])
            print(
                json.dumps(
                    {
                        "event": "action_chunk_compare_progress",
                        "model": name,
                        "windows": sample_count,
                        "first_slot_accuracy": first["accuracy"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    metrics = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(audit.get("checkpoint_step", -1)),
        "policy_horizon": policy_horizon,
        "backbone_action_tokens": backbone_action_tokens,
        "windows": sample_count,
        "actions": action_count,
        "ce": safe_div(ce_sum, action_count),
        "sequence_exact_match": safe_div(exact_sequences, sample_count),
        "first_slot": confusion_metrics(confusion_by_horizon[0]),
        "all_slots": confusion_metrics(all_confusion),
        "by_horizon": {
            str(horizon): confusion_metrics(confusion_by_horizon[horizon])
            for horizon in range(policy_horizon)
        },
        "by_history_micro": {
            str(history): {
                "count": int(values["count"]),
                "accuracy": safe_div(values["correct"], values["count"]),
            }
            for history, values in sorted(by_history.items())
        },
        "seconds_total": elapsed,
        "seconds_per_window": safe_div(elapsed, sample_count),
        "model_structure": model.structural_report(),
        "data_structure": builder.summary(),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, examples, sample_keys, torch.cat(predictions, dim=0)


def plot_results(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old = summary["models"]["old_h1"]["first_slot"]
    new = summary["models"]["new_h4"]["first_slot"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metrics, title in zip(axes, (old, new), ("Old H1 step1400", "New H4 step200 / slot 0")):
        matrix = np.asarray(metrics["confusion"], dtype=np.float64)
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_xticks(range(4), CLASS_NAMES, rotation=25, ha="right")
        ax.set_yticks(range(4), CLASS_NAMES)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground truth")
        ax.set_title(f"{title}\nacc={metrics['accuracy']:.3f}, macro R={metrics['macro_recall']:.3f}")
        for row in range(4):
            for col in range(4):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Row-normalized ratio")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--save-examples", type=int, default=64)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", type=Path, default=ROOT / "result/v1_stage3_action_chunk_compare")
    args = parser.parse_args()
    if args.num_windows < 1 or args.num_windows % args.batch_size != 0:
        raise SystemExit("num-windows must be positive and divisible by batch-size")

    install_non_reentrant_wan_checkpoint()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    old_checkpoint = args.old_checkpoint.resolve()
    new_checkpoint = args.new_checkpoint.resolve()
    new_config = adjacent_run_config(new_checkpoint)
    run_name = args.run_name or f"old{old_checkpoint.stem}_vs_new{new_checkpoint.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    models: dict[str, Any] = {}
    examples: dict[str, Any] = {}
    keys_by_model: dict[str, list[tuple[str, int]]] = {}
    preds_by_model: dict[str, torch.Tensor] = {}
    for name, checkpoint in (("old_h1", old_checkpoint), ("new_h4", new_checkpoint)):
        metrics, rows, keys, predictions = evaluate_checkpoint(
            name=name,
            checkpoint=checkpoint,
            config_payload=new_config["r2r_data"],
            num_windows=args.num_windows,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            dtype=dtype,
            save_examples=args.save_examples,
        )
        models[name] = metrics
        examples[name] = rows
        keys_by_model[name] = keys
        preds_by_model[name] = predictions

    if keys_by_model["old_h1"] != keys_by_model["new_h4"]:
        raise RuntimeError("paired evaluator sampled different windows for old/new checkpoints")
    old_first = preds_by_model["old_h1"][:, 0]
    new_first = preds_by_model["new_h4"][:, 0]
    summary = {
        "event": "stage3_h1_h4_paired_full_history_natural_eval",
        "protocol": {
            "dataset": "R2R train latent windows",
            "sampling": "paired full-prefix natural shuffled windows without balancing",
            "instruction": "correct instruction embedding",
            "num_windows": args.num_windows,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "old_target": "first future primitive only",
            "new_target": "four future primitives; slot 0 is the paired comparison",
            "model_forward": "complete Register + canonical shared Wan + policy head",
        },
        "models": models,
        "paired_first_slot": {
            "accuracy_delta_new_minus_old": (
                models["new_h4"]["first_slot"]["accuracy"]
                - models["old_h1"]["first_slot"]["accuracy"]
            ),
            "macro_recall_delta_new_minus_old": (
                models["new_h4"]["first_slot"]["macro_recall"]
                - models["old_h1"]["first_slot"]["macro_recall"]
            ),
            "prediction_disagreement_ratio": float((old_first != new_first).float().mean().item()),
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    plot_results(summary, output_dir / "paired_first_slot_confusion.png")
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
