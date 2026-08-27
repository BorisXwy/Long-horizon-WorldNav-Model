#!/usr/bin/env python3
"""Evaluate the full single-action Stage3 R2R policy in teacher-forced open loop.

The evaluator loads the complete shared Wan/Register checkpoint and its trained
four-class policy head.  It reports both the natural R2R window distribution
and an exactly balanced four-class distribution, with paired correct,
shuffled, and empty instruction conditions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from eval_v1_stage3_open_loop_policy import InstructionAblator  # noqa: E402
from nav.v1.data import (  # noqa: E402
    ACTION_NAMES,
    BalancedSingleActionR2RBatchBuilder,
    R2RStage3DataConfig,
    R2RStage3PolicyBatchBuilder,
    build_data_loader,
    r2r_data_config_from_dict,
    vln_action_to_combo,
)
from nav.v1.model import SingleActionStage3Model, build_model  # noqa: E402
from train_v1_stage2_final_cotrain import install_non_reentrant_wan_checkpoint, torch_dtype  # noqa: E402


CLASS_IDS = tuple(sorted(ACTION_NAMES))
CLASS_NAMES = [ACTION_NAMES[index] for index in CLASS_IDS]
COMBO_TO_CLASS = {vln_action_to_combo(index): index for index in CLASS_IDS}


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
    if not isinstance(model, SingleActionStage3Model) or assembly.source_payload is None:
        raise TypeError(type(model))
    return model, assembly.source_payload


def clone_data_config(payload: dict[str, Any], *, seed: int) -> R2RStage3DataConfig:
    cfg = r2r_data_config_from_dict(payload)
    cfg.action_horizon = 1
    cfg.batch_size = 1
    cfg.action_oversample_mode = "none"
    cfg.seed = seed
    return cfg


def make_builder(
    mode: str,
    payload: dict[str, Any],
    *,
    seed: int,
) -> tuple[R2RStage3PolicyBatchBuilder, R2RStage3DataConfig]:
    cfg = clone_data_config(payload, seed=seed)
    if mode == "natural":
        builder = build_data_loader("stage3_r2r_natural", config=cfg)
        if not isinstance(builder, R2RStage3PolicyBatchBuilder):
            raise TypeError(type(builder))
        return builder, cfg
    if mode == "balanced":
        builder = build_data_loader("stage3_r2r_balanced_single_action", config=cfg)
        if not isinstance(builder, BalancedSingleActionR2RBatchBuilder):
            raise TypeError(type(builder))
        return builder, cfg
    raise ValueError(mode)


def attach_single_action_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    combo = batch["action_combo"][:, :1].long()
    target = torch.full_like(combo, -1)
    for combo_id, class_id in COMBO_TO_CLASS.items():
        target[combo == combo_id] = class_id
    if bool((target < 0).any()):
        raise RuntimeError(f"unsupported R2R combo target: {combo.detach().cpu().tolist()}")
    batch["action_class"] = target
    batch["a_noise"] = torch.zeros(
        target.shape[0], 1, 6, device=target.device, dtype=batch["z_obs"].dtype
    )
    batch["action_timestep"] = torch.zeros(target.shape[0], device=target.device, dtype=torch.float32)
    return target


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def plot_confusion(matrix: np.ndarray, *, title: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sum, 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(4), CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for row in range(4):
        for col in range(4):
            value = normalized[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}\n(n={int(matrix[row, col])})",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Row-normalized ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_mode(
    *,
    model: SingleActionStage3Model,
    checkpoint_data_cfg: dict[str, Any],
    sampling_mode: str,
    instruction_mode: str,
    num_batches: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    save_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor, torch.Tensor, list[str], np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    builder, cfg = make_builder(sampling_mode, checkpoint_data_cfg, seed=seed)
    ablator = InstructionAblator(cfg, builder, seed=seed + 991)
    dataset_summary = builder.summary()

    total_ce = 0.0
    total = 0
    confusion = np.zeros((4, 4), dtype=np.int64)
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    sample_ids: list[str] = []
    examples: list[dict[str, Any]] = []
    by_history: dict[int, Counter[str]] = defaultdict(Counter)
    started = time.perf_counter()

    for batch_index in range(num_batches):
        batch, meta = builder.next_batch(device=device, dtype=dtype)
        target = attach_single_action_target(batch)
        batch, text_sources = ablator.apply(
            batch,
            meta,
            mode=instruction_mode,
            device=device,
            dtype=dtype,
        )
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            out = model.forward_policy(batch)
        logits = out["action_logits"].float()
        pred = logits.argmax(dim=-1)
        prob = logits.softmax(dim=-1)
        ce = F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1), reduction="sum")
        total_ce += float(ce.item())
        total += int(target.numel())
        history_micro = int(meta["history_micro"])
        by_history[history_micro]["count"] += int(target.numel())
        by_history[history_micro]["correct"] += int((pred == target).sum().item())

        gt_value = int(target.item())
        pred_value = int(pred.item())
        confusion[gt_value, pred_value] += 1
        predictions.append(pred.detach().cpu())
        probabilities.append(prob.detach().cpu())
        sample_ids.extend(meta["sample_ids"])
        if len(examples) < save_examples:
            examples.append(
                {
                    "sample_id": meta["sample_ids"][0],
                    "history_micro": history_micro,
                    "terminal": bool(meta["terminal_count"] > 0),
                    "instruction_mode": instruction_mode,
                    "text_source": text_sources[0],
                    "gt_class": gt_value,
                    "gt_name": ACTION_NAMES[gt_value],
                    "pred_class": pred_value,
                    "pred_name": ACTION_NAMES[pred_value],
                    "probabilities": [float(value) for value in prob[0, 0].cpu().tolist()],
                }
            )
        if (batch_index + 1) % 64 == 0:
            print(
                json.dumps(
                    {
                        "event": "single_action_eval_progress",
                        "sampling": sampling_mode,
                        "instruction": instruction_mode,
                        "batches": batch_index + 1,
                        "accuracy": safe_div(int(np.trace(confusion)), total),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    recall = np.diag(confusion) / np.maximum(support, 1)
    precision = np.diag(confusion) / np.maximum(predicted, 1)
    accuracy = safe_div(int(np.trace(confusion)), total)
    metrics = {
        "sampling_mode": sampling_mode,
        "instruction_mode": instruction_mode,
        "samples": total,
        "ce": safe_div(total_ce, total),
        "accuracy": accuracy,
        "macro_recall": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "balanced_random_ce": float(np.log(4.0)),
        "balanced_random_accuracy": 0.25,
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
        "by_history_micro": {
            str(key): {
                "count": int(value["count"]),
                "accuracy": safe_div(value["correct"], value["count"]),
            }
            for key, value in sorted(by_history.items())
        },
        "seconds_total": elapsed,
        "seconds_per_sample": safe_div(elapsed, total),
        "dataset_summary": dataset_summary,
    }
    return (
        metrics,
        examples,
        torch.cat(predictions, dim=0),
        torch.cat(probabilities, dim=0),
        sample_ids,
        confusion,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-batches", type=int, default=256)
    parser.add_argument("--sampling-modes", default="natural,balanced")
    parser.add_argument("--instruction-modes", default="correct,shuffled,empty")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--save-examples", type=int, default=64)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", type=Path, default=ROOT / "result/v1_stage3_single_action")
    args = parser.parse_args()

    install_non_reentrant_wan_checkpoint()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    checkpoint = args.checkpoint.resolve()
    model, payload = load_model(checkpoint, device=device, dtype=dtype)
    if "r2r_data_config" not in payload:
        raise RuntimeError("single-action checkpoint is missing r2r_data_config")

    sampling_modes = [value.strip() for value in args.sampling_modes.split(",") if value.strip()]
    instruction_modes = [value.strip() for value in args.instruction_modes.split(",") if value.strip()]
    if not sampling_modes or any(value not in {"natural", "balanced"} for value in sampling_modes):
        raise ValueError(f"bad sampling modes: {sampling_modes}")
    if not instruction_modes or any(value not in {"correct", "shuffled", "empty"} for value in instruction_modes):
        raise ValueError(f"bad instruction modes: {instruction_modes}")

    run_name = args.run_name or f"step{int(payload.get('step', -1)):06d}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "event": "stage3_single_action_open_loop_eval",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "source_checkpoint": payload.get("source_checkpoint"),
        "device": str(device),
        "dtype": args.dtype,
        "num_batches_per_mode": args.num_batches,
        "seed": args.seed,
        "sampling_modes": sampling_modes,
        "instruction_modes": instruction_modes,
        "model_structure": model.structural_report(),
        "modes": {},
        "instruction_sensitivity": {},
    }
    cached: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, list[str]]] = {}
    all_examples: dict[str, Any] = {}

    for sampling_mode in sampling_modes:
        for instruction_mode in instruction_modes:
            key = f"{sampling_mode}__{instruction_mode}"
            metrics, examples, pred, prob, sample_ids, confusion = evaluate_mode(
                model=model,
                checkpoint_data_cfg=payload["r2r_data_config"],
                sampling_mode=sampling_mode,
                instruction_mode=instruction_mode,
                num_batches=args.num_batches,
                seed=args.seed,
                device=device,
                dtype=dtype,
                save_examples=args.save_examples,
            )
            summary["modes"][key] = metrics
            cached[(sampling_mode, instruction_mode)] = (pred, prob, sample_ids)
            all_examples[key] = examples
            plot_confusion(
                confusion,
                title=f"Stage3 single action | {sampling_mode} | {instruction_mode}",
                output_path=output_dir / f"confusion_{key}.png",
            )

        if "correct" in instruction_modes:
            ref_pred, ref_prob, ref_ids = cached[(sampling_mode, "correct")]
            summary["instruction_sensitivity"][sampling_mode] = {}
            for instruction_mode in instruction_modes:
                if instruction_mode == "correct":
                    continue
                pred, prob, sample_ids = cached[(sampling_mode, instruction_mode)]
                if sample_ids != ref_ids:
                    raise RuntimeError(f"paired sample mismatch for {sampling_mode}/{instruction_mode}")
                total_variation = 0.5 * (ref_prob.float() - prob.float()).abs().sum(dim=-1)
                kl = (
                    ref_prob.float().clamp_min(1e-8)
                    * (
                        ref_prob.float().clamp_min(1e-8).log()
                        - prob.float().clamp_min(1e-8).log()
                    )
                ).sum(dim=-1)
                summary["instruction_sensitivity"][sampling_mode][instruction_mode] = {
                    "paired_samples": len(ref_ids),
                    "top1_disagreement_ratio": float((ref_pred != pred).float().mean().item()),
                    "probability_total_variation_mean": float(total_variation.mean().item()),
                    "kl_correct_to_ablation_mean": float(kl.mean().item()),
                }

    if device.type == "cuda":
        summary["cuda_max_memory_gb"] = float(torch.cuda.max_memory_reserved(device) / 1024**3)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "examples.json").write_text(json.dumps(all_examples, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
