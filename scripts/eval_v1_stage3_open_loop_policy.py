#!/usr/bin/env python3
"""Evaluate the current formal Stage3 R2R policy in open loop.

The evaluator reuses the full Stage3 model and the current R2R future-action
window builder:

``history_latents + A_hist -> Register; Register + Z_obs + instruction + A_noise -> combo_logits``

It reports action accuracy on teacher-forced R2R train observations and writes
open-loop sequence, horizon-accuracy, and confusion-matrix figures.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
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

from nav.v1.stage2_final.data import ACTION_BASE, COMBO_DIM  # noqa: E402
from nav.v1.stage3_r2r import (  # noqa: E402
    R2RStage3DataConfig,
    R2RStage3PolicyBatchBuilder,
    combo_to_name,
    vln_action_to_combo,
)
from train_v1_stage2_final_cotrain import install_non_reentrant_wan_checkpoint, torch_dtype  # noqa: E402
from train_v1_stage3_r2r_future_action import load_stage2_model, make_dataclass  # noqa: E402


ACTION_IDS = [
    vln_action_to_combo(0),  # STOP
    vln_action_to_combo(1),  # MOVE_FORWARD
    vln_action_to_combo(2),  # TURN_LEFT
    vln_action_to_combo(3),  # TURN_RIGHT
]
ACTION_NAMES = [combo_to_name(value) for value in ACTION_IDS]
ACTION_SHORT = {
    "STOP": "S",
    "MOVE_FORWARD": "M",
    "TURN_LEFT": "L",
    "TURN_RIGHT": "R",
}
ACTION_COLORS = {
    "STOP": "#7f7f7f",
    "MOVE_FORWARD": "#2ca02c",
    "TURN_LEFT": "#1f77b4",
    "TURN_RIGHT": "#ff7f0e",
    "OTHER": "#d62728",
}


def combo_to_pair(combo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    combo = combo.long().clamp_min(0).clamp_max(COMBO_DIM - 1)
    return combo // ACTION_BASE, combo % ACTION_BASE


def action_name(combo_id: int) -> str:
    return combo_to_name(int(combo_id))


def counter_to_named(counter: Counter[int], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"combo": int(key), "name": action_name(int(key)), "count": int(value)}
        for key, value in counter.most_common(limit)
    ]


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


class InstructionAblator:
    """Construct paired correct/shuffled/empty instruction conditions."""

    def __init__(self, cfg: R2RStage3DataConfig, builder: R2RStage3PolicyBatchBuilder, *, seed: int) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.candidates = [(window.dataset, window.sample_id) for window in builder.windows]
        self.cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        empty = torch.load(cfg.text_empty, map_location="cpu", weights_only=False)
        self.empty_y = empty["y"].to(torch.bfloat16)
        self.empty_mask = empty["y_mask"]

    def _cached_text(self, dataset: str, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = f"{dataset}/{sample_id}"
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        path = self.cfg.text_cache_root / dataset / f"{sample_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing shuffled instruction embedding: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = (payload["y"].to(torch.bfloat16), payload["y_mask"])
        self.cache[key] = value
        while len(self.cache) > max(1, int(self.cfg.text_cache_items)):
            self.cache.popitem(last=False)
        return value

    def apply(
        self,
        batch: dict[str, torch.Tensor],
        meta: dict[str, Any],
        *,
        mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        if mode == "correct":
            return batch, list(meta["text_sources"])
        ys: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        sources: list[str] = []
        if mode == "empty":
            for _ in meta["sample_ids"]:
                ys.append(self.empty_y)
                masks.append(self.empty_mask)
                sources.append(str(self.cfg.text_empty))
        elif mode == "shuffled":
            for current_sample_id in meta["sample_ids"]:
                dataset, sample_id = self.rng.choice(self.candidates)
                while sample_id == current_sample_id:
                    dataset, sample_id = self.rng.choice(self.candidates)
                y, y_mask = self._cached_text(dataset, sample_id)
                ys.append(y)
                masks.append(y_mask)
                sources.append(f"shuffled:{dataset}/{sample_id}")
        else:
            raise ValueError(mode)
        out = dict(batch)
        out["y"] = torch.cat(ys, dim=0).to(device=device, dtype=dtype, non_blocking=True)
        out["y_mask"] = torch.cat(masks, dim=0).to(device=device, dtype=dtype, non_blocking=True)
        return out, sources


def plot_open_loop_sequences(
    examples: list[dict[str, Any]],
    *,
    checkpoint_step: int,
    mode: str,
    output_path: Path,
    max_examples: int = 16,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    rows = examples[:max_examples]
    if not rows:
        return
    horizon = len(rows[0]["gt_name"])
    fig_height = max(4.0, 0.72 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(max(12.0, horizon * 0.85 + 5.0), fig_height))
    ax.set_xlim(-4.2, horizon)
    ax.set_ylim(0, len(rows) * 2)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(horizon) + 0.5, [f"t+{index + 1}" for index in range(horizon)])
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.set_title(
        f"R2R train open-loop action chunks | checkpoint step {checkpoint_step} | {mode}",
        pad=28,
    )

    for row_index, row in enumerate(rows):
        gt_names = row["gt_name"]
        pred_names = [name if name in ACTION_SHORT else "OTHER" for name in row["pred_name"]]
        sample_label = str(row["sample_id"])
        if len(sample_label) > 30:
            sample_label = sample_label[:27] + "..."
        ax.text(-4.1, row_index * 2 + 0.5, f"{sample_label}\nK={row['history_micro']}", va="center", fontsize=7)
        ax.text(-0.15, row_index * 2 + 0.35, "GT", ha="right", va="center", fontsize=7, fontweight="bold")
        ax.text(-0.15, row_index * 2 + 1.35, "Pred", ha="right", va="center", fontsize=7, fontweight="bold")
        for horizon_index, (gt_name, pred_name) in enumerate(zip(gt_names, pred_names)):
            gt_color = ACTION_COLORS.get(gt_name, ACTION_COLORS["OTHER"])
            pred_color = ACTION_COLORS.get(pred_name, ACTION_COLORS["OTHER"])
            ax.add_patch(Rectangle((horizon_index, row_index * 2), 1, 0.7, facecolor=gt_color, edgecolor="white"))
            ax.add_patch(
                Rectangle(
                    (horizon_index, row_index * 2 + 1),
                    1,
                    0.7,
                    facecolor=pred_color,
                    edgecolor=("black" if pred_name != gt_name else "white"),
                    linewidth=(1.4 if pred_name != gt_name else 0.6),
                )
            )
            ax.text(horizon_index + 0.5, row_index * 2 + 0.35, ACTION_SHORT.get(gt_name, "?"), ha="center", va="center", color="white", fontweight="bold", fontsize=8)
            ax.text(horizon_index + 0.5, row_index * 2 + 1.35, ACTION_SHORT.get(pred_name, "?"), ha="center", va="center", color="white", fontweight="bold", fontsize=8)

    handles = [Patch(color=ACTION_COLORS[name], label=name) for name in [*ACTION_NAMES, "OTHER"]]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_accuracy(mode_payloads: dict[str, dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for mode, payload in mode_payloads.items():
        by_horizon = payload["by_horizon"]
        xs = [int(key) + 1 for key in sorted(by_horizon, key=lambda value: int(value))]
        axes[0].plot(xs, [by_horizon[str(x - 1)]["combo_acc"] for x in xs], marker="o", label=f"{mode}: combo")
        axes[0].plot(xs, [by_horizon[str(x - 1)]["trans_acc"] for x in xs], linestyle="--", alpha=0.75, label=f"{mode}: trans")
        axes[0].plot(xs, [by_horizon[str(x - 1)]["rot_acc"] for x in xs], linestyle=":", alpha=0.75, label=f"{mode}: rot")
    axes[0].set_xlabel("Future action horizon")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    modes = list(mode_payloads)
    x = np.arange(len(modes))
    width = 0.19
    series = [
        ("combo_top1_acc", "Combo"),
        ("trans_acc", "Translation"),
        ("rot_acc", "Rotation"),
        ("macro_recall", "Macro recall"),
    ]
    for index, (key, label) in enumerate(series):
        values = [mode_payloads[mode][key] for mode in modes]
        axes[1].bar(x + (index - 1.5) * width, values, width=width, label=label)
    majority = [mode_payloads[mode]["majority_baseline_acc"] for mode in modes]
    axes[1].scatter(x, majority, marker="x", color="black", s=65, label="Majority baseline", zorder=5)
    axes[1].set_xticks(x, modes)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Accuracy / recall")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("R2R train open-loop policy metrics")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(confusion: Counter[tuple[int, int]], *, mode: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.zeros((len(ACTION_IDS), len(ACTION_IDS) + 1), dtype=np.float64)
    id_to_col = {value: index for index, value in enumerate(ACTION_IDS)}
    for (gt, pred), count in confusion.items():
        if gt not in id_to_col:
            continue
        matrix[id_to_col[gt], id_to_col.get(pred, len(ACTION_IDS))] += count
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(ACTION_IDS) + 1), [*ACTION_NAMES, "OTHER"], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(ACTION_IDS)), ACTION_NAMES)
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("Ground-truth action")
    ax.set_title(f"R2R train open-loop normalized confusion | {mode}")
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            ax.text(col, row, f"{normalized[row, col]:.2f}\n(n={int(matrix[row, col])})", ha="center", va="center", fontsize=8, color=("white" if normalized[row, col] > 0.55 else "black"))
    fig.colorbar(image, ax=ax, label="Row-normalized ratio")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-batches", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--history-micro-choices", default="")
    parser.add_argument("--max-r2r-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--action-noise-mode", choices=("random", "zero", "both"), default="both")
    parser.add_argument(
        "--instruction-mode",
        choices=("correct", "shuffled", "empty", "all"),
        default="correct",
    )
    parser.add_argument("--latent-manifest-dir", type=Path, default=None)
    parser.add_argument("--rendered-manifest", type=Path, default=None)
    parser.add_argument("--text-cache-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "result/v1_stage3_open_loop_policy")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--save-examples", type=int, default=64)
    parser.add_argument("--plot-examples", type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_non_reentrant_wan_checkpoint()

    checkpoint = args.checkpoint.resolve()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    model, source = load_stage2_model(checkpoint, device=device, dtype=dtype)
    model.eval()
    if "r2r_data_config" not in source:
        raise RuntimeError(
            "checkpoint is not from scripts/train_v1_stage3_r2r_future_action.py: "
            "missing r2r_data_config"
        )
    r2r_cfg = make_dataclass(R2RStage3DataConfig, source["r2r_data_config"])
    r2r_cfg.batch_size = args.batch_size
    r2r_cfg.seed = args.seed
    if args.history_micro_choices:
        r2r_cfg.history_micro_choices = args.history_micro_choices
    if args.max_r2r_episodes > 0:
        r2r_cfg.max_episodes = args.max_r2r_episodes
    if args.latent_manifest_dir is not None:
        r2r_cfg.latent_manifest_dir = args.latent_manifest_dir
    if args.rendered_manifest is not None:
        r2r_cfg.rendered_manifest = args.rendered_manifest
    if args.text_cache_root is not None:
        r2r_cfg.text_cache_root = args.text_cache_root

    def make_builder() -> R2RStage3PolicyBatchBuilder:
        return R2RStage3PolicyBatchBuilder(r2r_cfg)

    dataset_summary = make_builder().summary()
    action_modes = ["random", "zero"] if args.action_noise_mode == "both" else [args.action_noise_mode]
    instruction_modes = ["correct", "shuffled", "empty"] if args.instruction_mode == "all" else [args.instruction_mode]
    eval_specs = [
        (
            action_mode if instruction_modes == ["correct"] else f"{instruction_mode}__{action_mode}_anoise",
            instruction_mode,
            action_mode,
        )
        for instruction_mode in instruction_modes
        for action_mode in action_modes
    ]
    run_name = args.run_name or f"r2r_train_step{int(source.get('step', -1)):06d}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.out_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "event": "stage3_r2r_train_open_loop_eval",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(source.get("step", -1)),
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": args.dtype,
        "num_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "history_micro_choices": r2r_cfg.history_micro_choices,
        "instruction_modes": instruction_modes,
        "action_noise_modes": action_modes,
        "dataset_split": "R2R train teacher-forced observations",
        "dataset_summary": dataset_summary,
        "modes": {},
    }
    examples_by_mode: dict[str, list[dict[str, Any]]] = {mode_key: [] for mode_key, _, _ in eval_specs}
    confusions: dict[str, Counter[tuple[int, int]]] = {}
    predictions_by_mode: dict[str, list[torch.Tensor]] = defaultdict(list)
    probabilities_by_mode: dict[str, list[torch.Tensor]] = defaultdict(list)
    sample_ids_by_mode: dict[str, list[str]] = defaultdict(list)

    with torch.no_grad():
        for mode_key, instruction_mode, action_mode in eval_specs:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            builder = make_builder()
            instruction_ablator = InstructionAblator(r2r_cfg, builder, seed=args.seed + 991)
            total_valid = total_ce = 0.0
            correct_combo = correct_trans = correct_rot = exact_sequences = 0.0
            sequence_count = 0.0
            other_predictions = 0.0
            by_horizon: dict[int, dict[str, float]] = defaultdict(lambda: {"valid": 0.0, "combo_correct": 0.0, "trans_correct": 0.0, "rot_correct": 0.0})
            by_history: dict[int, dict[str, float]] = defaultdict(lambda: {"valid": 0.0, "combo_correct": 0.0, "ce_sum": 0.0, "samples": 0.0})
            gt_counter: Counter[int] = Counter()
            pred_counter: Counter[int] = Counter()
            confusion: Counter[tuple[int, int]] = Counter()
            per_gt: dict[int, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "correct": 0.0})
            started = time.time()

            for batch_index in range(args.num_batches):
                batch, meta = builder.next_batch(device=device, dtype=dtype)
                batch, effective_text_sources = instruction_ablator.apply(
                    batch,
                    meta,
                    mode=instruction_mode,
                    device=device,
                    dtype=dtype,
                )
                if action_mode == "zero":
                    batch["a_noise"].zero_()
                    batch["action_timestep"].zero_()
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                    out = model.forward_stage3_policy(batch, lambda_ce=1.0)
                logits = out["combo_logits"].float()
                target = batch["action_combo"].long()
                mask = batch["action_loss_mask"].float()
                pred = logits.argmax(dim=-1)
                probabilities = logits.softmax(dim=-1)
                predictions_by_mode[mode_key].append(pred.detach().cpu())
                probabilities_by_mode[mode_key].append(probabilities.detach().cpu())
                sample_ids_by_mode[mode_key].extend(meta["sample_ids"])
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none").view_as(target)
                valid = mask > 0
                valid_count = float(valid.float().sum().item())
                if valid_count <= 0:
                    continue
                combo_ok = (pred == target) & valid
                gt_trans, gt_rot = combo_to_pair(target)
                pred_trans, pred_rot = combo_to_pair(pred)
                trans_ok = (gt_trans == pred_trans) & valid
                rot_ok = (gt_rot == pred_rot) & valid
                total_valid += valid_count
                total_ce += float((ce * mask).sum().item())
                correct_combo += float(combo_ok.float().sum().item())
                correct_trans += float(trans_ok.float().sum().item())
                correct_rot += float(rot_ok.float().sum().item())
                sequence_valid = valid.sum(dim=1) > 0
                sequence_exact = (combo_ok | ~valid).all(dim=1) & sequence_valid
                exact_sequences += float(sequence_exact.float().sum().item())
                sequence_count += float(sequence_valid.float().sum().item())
                known_pred = torch.zeros_like(pred, dtype=torch.bool)
                for action_id in ACTION_IDS:
                    known_pred |= pred == action_id
                other_predictions += float((~known_pred & valid).float().sum().item())
                history_micro = int(meta["history_micro"])
                by_history[history_micro]["valid"] += valid_count
                by_history[history_micro]["combo_correct"] += float(combo_ok.float().sum().item())
                by_history[history_micro]["ce_sum"] += float((ce * mask).sum().item())
                by_history[history_micro]["samples"] += float(target.shape[0])
                for horizon_index in range(target.shape[1]):
                    horizon_valid = valid[:, horizon_index]
                    horizon_count = float(horizon_valid.float().sum().item())
                    if horizon_count <= 0:
                        continue
                    by_horizon[horizon_index]["valid"] += horizon_count
                    by_horizon[horizon_index]["combo_correct"] += float(combo_ok[:, horizon_index].float().sum().item())
                    by_horizon[horizon_index]["trans_correct"] += float(trans_ok[:, horizon_index].float().sum().item())
                    by_horizon[horizon_index]["rot_correct"] += float(rot_ok[:, horizon_index].float().sum().item())
                for gt, predicted, ok in zip(target[valid].cpu().tolist(), pred[valid].cpu().tolist(), combo_ok[valid].cpu().tolist()):
                    gt_counter[int(gt)] += 1
                    pred_counter[int(predicted)] += 1
                    confusion[(int(gt), int(predicted))] += 1
                    per_gt[int(gt)]["count"] += 1.0
                    per_gt[int(gt)]["correct"] += float(bool(ok))
                if len(examples_by_mode[mode_key]) < args.save_examples:
                    top_probability = probabilities.gather(-1, pred[..., None]).squeeze(-1)
                    for item_index in range(target.shape[0]):
                        examples_by_mode[mode_key].append(
                            {
                                "sample_id": meta["sample_ids"][item_index],
                                "history_micro": history_micro,
                                "label_start_action": int(meta["label_start_actions"][item_index]),
                                "terminal": bool(meta["terminal_count"] > 0),
                                "instruction_mode": instruction_mode,
                                "text_source": effective_text_sources[item_index],
                                "gt_combo": [int(value) for value in target[item_index].cpu().tolist()],
                                "gt_name": [action_name(int(value)) for value in target[item_index].cpu().tolist()],
                                "pred_combo": [int(value) for value in pred[item_index].cpu().tolist()],
                                "pred_name": [action_name(int(value)) for value in pred[item_index].cpu().tolist()],
                                "valid_mask": [float(value) for value in mask[item_index].cpu().tolist()],
                                "pred_top_probability": [float(value) for value in top_probability[item_index].cpu().tolist()],
                            }
                        )
                        if len(examples_by_mode[mode_key]) >= args.save_examples:
                            break
                if (batch_index + 1) % 32 == 0:
                    print(json.dumps({"event": "stage3_open_loop_progress", "mode": mode_key, "batches": batch_index + 1, "combo_acc": safe_div(correct_combo, total_valid)}, ensure_ascii=False), flush=True)

            per_class_recall = [
                {
                    "combo": action_id,
                    "name": action_name(action_id),
                    "count": int(per_gt[action_id]["count"]),
                    "recall": safe_div(per_gt[action_id]["correct"], per_gt[action_id]["count"]),
                }
                for action_id in ACTION_IDS
                if per_gt[action_id]["count"] > 0
            ]
            majority = gt_counter.most_common(1)[0] if gt_counter else (None, 0)
            elapsed = time.time() - started
            mode_payload = {
                "instruction_mode": instruction_mode,
                "action_noise_mode": action_mode,
                "samples": int(sequence_count),
                "valid_actions": total_valid,
                "ce": safe_div(total_ce, total_valid),
                "combo_top1_acc": safe_div(correct_combo, total_valid),
                "trans_acc": safe_div(correct_trans, total_valid),
                "rot_acc": safe_div(correct_rot, total_valid),
                "sequence_exact_match": safe_div(exact_sequences, sequence_count),
                "other_prediction_ratio": safe_div(other_predictions, total_valid),
                "macro_recall": float(np.mean([row["recall"] for row in per_class_recall])) if per_class_recall else 0.0,
                "majority_baseline_combo": None if majority[0] is None else int(majority[0]),
                "majority_baseline_name": None if majority[0] is None else action_name(int(majority[0])),
                "majority_baseline_acc": safe_div(float(majority[1]), total_valid),
                "by_horizon": {
                    str(index): {
                        "valid": row["valid"],
                        "combo_acc": safe_div(row["combo_correct"], row["valid"]),
                        "trans_acc": safe_div(row["trans_correct"], row["valid"]),
                        "rot_acc": safe_div(row["rot_correct"], row["valid"]),
                    }
                    for index, row in sorted(by_horizon.items())
                },
                "by_history_micro": {
                    str(history): {
                        "samples": int(row["samples"]),
                        "valid": row["valid"],
                        "ce": safe_div(row["ce_sum"], row["valid"]),
                        "combo_acc": safe_div(row["combo_correct"], row["valid"]),
                    }
                    for history, row in sorted(by_history.items())
                },
                "per_gt_recall": per_class_recall,
                "gt_distribution": counter_to_named(gt_counter),
                "pred_distribution": counter_to_named(pred_counter),
                "confusion_top": [
                    {
                        "gt_combo": int(gt),
                        "gt_name": action_name(int(gt)),
                        "pred_combo": int(predicted),
                        "pred_name": action_name(int(predicted)),
                        "count": int(count),
                    }
                    for (gt, predicted), count in confusion.most_common(30)
                ],
                "elapsed_sec": elapsed,
                "sec_per_batch": elapsed / max(args.num_batches, 1),
            }
            summary["modes"][mode_key] = mode_payload
            confusions[mode_key] = confusion

    instruction_sensitivity: dict[str, Any] = {}
    if "correct" in instruction_modes:
        for action_mode in action_modes:
            reference_key = (
                action_mode
                if instruction_modes == ["correct"]
                else f"correct__{action_mode}_anoise"
            )
            reference_pred = torch.cat(predictions_by_mode[reference_key], dim=0)
            reference_prob = torch.cat(probabilities_by_mode[reference_key], dim=0).float()
            instruction_sensitivity[action_mode] = {}
            for instruction_mode in instruction_modes:
                if instruction_mode == "correct":
                    continue
                compared_key = f"{instruction_mode}__{action_mode}_anoise"
                if sample_ids_by_mode[compared_key] != sample_ids_by_mode[reference_key]:
                    raise RuntimeError(
                        f"paired instruction ablation sampled different windows: {reference_key} vs {compared_key}"
                    )
                compared_pred = torch.cat(predictions_by_mode[compared_key], dim=0)
                compared_prob = torch.cat(probabilities_by_mode[compared_key], dim=0).float()
                total_variation = 0.5 * (reference_prob - compared_prob).abs().sum(dim=-1)
                kl = (
                    reference_prob.clamp_min(1e-8)
                    * (
                        reference_prob.clamp_min(1e-8).log()
                        - compared_prob.clamp_min(1e-8).log()
                    )
                ).sum(dim=-1)
                instruction_sensitivity[action_mode][instruction_mode] = {
                    "paired_samples": len(sample_ids_by_mode[reference_key]),
                    "paired_actions": int(reference_pred.numel()),
                    "top1_disagreement_ratio": float((reference_pred != compared_pred).float().mean().item()),
                    "probability_total_variation_mean": float(total_variation.mean().item()),
                    "kl_correct_to_ablation_mean": float(kl.mean().item()),
                }
    summary["instruction_sensitivity"] = instruction_sensitivity
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "examples.json").write_text(json.dumps(examples_by_mode, ensure_ascii=False, indent=2) + "\n")
    plot_horizon_accuracy(summary["modes"], output_dir / "open_loop_metrics.png")
    for mode_key, _, _ in eval_specs:
        plot_open_loop_sequences(
            examples_by_mode[mode_key],
            checkpoint_step=int(source.get("step", -1)),
            mode=mode_key,
            output_path=output_dir / f"open_loop_sequences_{mode_key}.png",
            max_examples=args.plot_examples,
        )
        plot_confusion(
            confusions[mode_key],
            mode=mode_key,
            output_path=output_dir / f"confusion_{mode_key}.png",
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "modes": summary["modes"],
                "instruction_sensitivity": instruction_sensitivity,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
