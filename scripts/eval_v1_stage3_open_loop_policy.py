#!/usr/bin/env python3
"""Open-loop policy evaluation for the formal V1 Stage3 checkpoint.

This script intentionally reuses the full Stage3 model/data path instead of a
toy/scaffold path:

``history_latents + A_hist -> Register; Z_obs + instruction + A_noise -> shared Wan action tokens``

It evaluates whether the predicted ``combo_logits=[B,H,144]`` match the dataset
action labels under teacher-forced observations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nav.v1.stage2_final.data import ACTION_BASE, COMBO_DIM  # noqa: E402
from train_v1_stage2_final_cotrain import install_non_reentrant_wan_checkpoint, torch_dtype  # noqa: E402
from train_v1_stage3_final_vln_cotrain import VLNT4PolicyDataset, load_stage2_model, make_dataclass  # noqa: E402
from nav.v1.stage2_final import FinalStage2DataConfig  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "log/v1_stage3_final_vln_cotrain/stage3_final_vln_instr_from_stage2step2000_rxr_iw1_2k_20260821/checkpoints/step_000200.pt"


def combo_to_pair(combo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    combo = combo.long().clamp_min(0).clamp_max(COMBO_DIM - 1)
    return combo // ACTION_BASE, combo % ACTION_BASE


def action_name(combo_id: int) -> str:
    mapping = {
        10 * ACTION_BASE + 0: "STOP",
        1 * ACTION_BASE + 0: "MOVE_FORWARD",
        0 * ACTION_BASE + 3: "TURN_LEFT",
        0 * ACTION_BASE + 4: "TURN_RIGHT",
    }
    return mapping.get(int(combo_id), f"combo_{int(combo_id)}")


def counter_to_named(counter: Counter[int], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"combo": int(k), "name": action_name(int(k)), "count": int(v)}
        for k, v in counter.most_common(limit)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=None, help="默认读取 checkpoint train_config.vln_manifest")
    parser.add_argument("--text-cache-root", type=Path, default=None, help="默认读取 checkpoint train_config.vln_text_cache_root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--history-iw-chunks", default=None)
    parser.add_argument("--max-vln-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--action-noise-mode", choices=("random", "zero", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "result/v1_stage3_open_loop_policy")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-examples", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_non_reentrant_wan_checkpoint()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_cfg = ckpt.get("train_config", {})
    data_cfg = make_dataclass(FinalStage2DataConfig, ckpt["data_config"])
    manifest = args.manifest or Path(train_cfg["vln_manifest"])
    text_cache_root = args.text_cache_root or Path(train_cfg["vln_text_cache_root"])
    history_iw_chunks = args.history_iw_chunks or train_cfg.get("history_iw_chunks", "1")
    require_text_cache = bool(train_cfg.get("require_vln_text_cache", True))

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    model, source = load_stage2_model(args.checkpoint, device=device, dtype=dtype)
    model.eval()

    def make_builder() -> VLNT4PolicyDataset:
        return VLNT4PolicyDataset(
            manifest=manifest,
            history_iw_chunks=history_iw_chunks,
            action_horizon=model.cfg.action_horizon,
            batch_size=args.batch_size,
            text_empty=data_cfg.text_empty,
            text_cache_root=text_cache_root,
            require_text_cache=require_text_cache,
            max_episodes=args.max_vln_episodes,
            seed=args.seed,
        )

    summary_builder = make_builder()

    modes = ["random", "zero"] if args.action_noise_mode == "both" else [args.action_noise_mode]
    run_name = args.run_name or f"stage3_open_loop_step{int(source.get('step', -1)):06d}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(source.get("step", -1)),
        "manifest": str(manifest),
        "text_cache_root": str(text_cache_root),
        "device": str(device),
        "dtype": args.dtype,
        "num_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "history_iw_chunks": history_iw_chunks,
        "dataset_summary": summary_builder.summary(),
        "modes": {},
    }
    examples_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}

    with torch.no_grad():
        for mode in modes:
            # Rebuild with the same seed for every mode so that random-vs-zero
            # action noise is compared on the same sampled open-loop windows.
            builder = make_builder()
            total_valid = 0.0
            total_ce = 0.0
            correct_combo = 0.0
            correct_trans = 0.0
            correct_rot = 0.0
            by_horizon: dict[int, dict[str, float]] = defaultdict(lambda: {"valid": 0.0, "combo_correct": 0.0, "trans_correct": 0.0, "rot_correct": 0.0})
            gt_counter: Counter[int] = Counter()
            pred_counter: Counter[int] = Counter()
            confusion: Counter[tuple[int, int]] = Counter()
            per_gt: dict[int, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "correct": 0.0})
            sample_count = 0
            t0 = time.time()

            for _ in range(args.num_batches):
                batch, meta = builder.next_batch(device=device, dtype=dtype)
                if mode == "zero":
                    batch["a_noise"].zero_()
                    batch["action_timestep"].zero_()
                out = model.forward_stage3_policy(batch, lambda_ce=1.0)
                logits = out["combo_logits"].float()
                target = batch["action_combo"].long()
                mask = batch["action_loss_mask"].float()
                pred = logits.argmax(dim=-1)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target.reshape(-1),
                    reduction="none",
                ).view_as(target)
                valid = mask > 0
                valid_count = float(valid.float().sum().item())
                if valid_count <= 0:
                    continue
                sample_count += int(target.shape[0])
                total_valid += valid_count
                total_ce += float((ce * mask).sum().item())
                combo_ok = (pred == target) & valid
                gt_trans, gt_rot = combo_to_pair(target)
                pr_trans, pr_rot = combo_to_pair(pred)
                trans_ok = (gt_trans == pr_trans) & valid
                rot_ok = (gt_rot == pr_rot) & valid
                correct_combo += float(combo_ok.float().sum().item())
                correct_trans += float(trans_ok.float().sum().item())
                correct_rot += float(rot_ok.float().sum().item())
                for h in range(target.shape[1]):
                    hv = valid[:, h]
                    h_valid = float(hv.float().sum().item())
                    if h_valid <= 0:
                        continue
                    by_horizon[h]["valid"] += h_valid
                    by_horizon[h]["combo_correct"] += float(combo_ok[:, h].float().sum().item())
                    by_horizon[h]["trans_correct"] += float(trans_ok[:, h].float().sum().item())
                    by_horizon[h]["rot_correct"] += float(rot_ok[:, h].float().sum().item())
                for gt, pr, ok in zip(target[valid].detach().cpu().tolist(), pred[valid].detach().cpu().tolist(), combo_ok[valid].detach().cpu().tolist()):
                    gt_counter[int(gt)] += 1
                    pred_counter[int(pr)] += 1
                    confusion[(int(gt), int(pr))] += 1
                    per_gt[int(gt)]["count"] += 1.0
                    per_gt[int(gt)]["correct"] += float(bool(ok))
                if len(examples_by_mode[mode]) < args.save_examples:
                    probs = logits.softmax(dim=-1)
                    top_prob = probs.gather(-1, pred[..., None]).squeeze(-1)
                    for b in range(target.shape[0]):
                        examples_by_mode[mode].append(
                            {
                                "sample_id": meta.get("sample_ids", [""])[b] if b < len(meta.get("sample_ids", [])) else "",
                                "history_iw": meta.get("history_iw"),
                                "text_source": meta.get("text_sources", [""])[b] if b < len(meta.get("text_sources", [])) else "",
                                "gt_combo": [int(x) for x in target[b].detach().cpu().tolist()],
                                "gt_name": [action_name(int(x)) for x in target[b].detach().cpu().tolist()],
                                "pred_combo": [int(x) for x in pred[b].detach().cpu().tolist()],
                                "pred_name": [action_name(int(x)) for x in pred[b].detach().cpu().tolist()],
                                "valid_mask": [float(x) for x in mask[b].detach().cpu().tolist()],
                                "pred_top_prob": [float(x) for x in top_prob[b].detach().cpu().tolist()],
                            }
                        )
                        if len(examples_by_mode[mode]) >= args.save_examples:
                            break

            elapsed = time.time() - t0
            horizon_payload = {}
            for h, stats in sorted(by_horizon.items()):
                denom = max(stats["valid"], 1.0)
                horizon_payload[str(h)] = {
                    "valid": stats["valid"],
                    "combo_acc": stats["combo_correct"] / denom,
                    "trans_acc": stats["trans_correct"] / denom,
                    "rot_acc": stats["rot_correct"] / denom,
                }
            majority = gt_counter.most_common(1)[0] if gt_counter else (None, 0)
            per_class_recall = [
                {
                    "combo": int(combo),
                    "name": action_name(int(combo)),
                    "count": int(stats["count"]),
                    "recall": stats["correct"] / max(stats["count"], 1.0),
                }
                for combo, stats in sorted(per_gt.items(), key=lambda item: item[1]["count"], reverse=True)
            ]
            summary["modes"][mode] = {
                "samples": sample_count,
                "valid_actions": total_valid,
                "ce": total_ce / max(total_valid, 1.0),
                "combo_top1_acc": correct_combo / max(total_valid, 1.0),
                "trans_acc": correct_trans / max(total_valid, 1.0),
                "rot_acc": correct_rot / max(total_valid, 1.0),
                "majority_baseline_combo": None if majority[0] is None else int(majority[0]),
                "majority_baseline_name": None if majority[0] is None else action_name(int(majority[0])),
                "majority_baseline_acc": float(majority[1]) / max(total_valid, 1.0),
                "by_horizon": horizon_payload,
                "per_gt_recall": per_class_recall,
                "gt_distribution_top": counter_to_named(gt_counter),
                "pred_distribution_top": counter_to_named(pred_counter),
                "confusion_top": [
                    {
                        "gt_combo": int(gt),
                        "gt_name": action_name(int(gt)),
                        "pred_combo": int(pr),
                        "pred_name": action_name(int(pr)),
                        "count": int(count),
                    }
                    for (gt, pr), count in confusion.most_common(30)
                ],
                "elapsed_sec": elapsed,
                "sec_per_batch": elapsed / max(args.num_batches, 1),
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "examples.json").write_text(json.dumps(examples_by_mode, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
