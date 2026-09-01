#!/usr/bin/env python3
"""Train the full V1 model on a discrete R2R action target.

The video/Register/pose path is the complete current Stage2 implementation.
The policy path reads one or more deterministic action-query hiddens from the
final shared Wan layer and decodes every query with the same FP32 four-class
MLP.  Both the historical balanced one-step protocol and the formal
full-history/natural-distribution action-chunk protocol remain addressable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nav.v1.data import (  # noqa: E402
    ACTION_NAMES,
    BalancedSingleActionR2RBatchBuilder,
    FinalStage2BatchBuilder,
    FinalStage2DataConfig,
    FullHistoryBalancedSingleActionR2RBatchBuilder,
    FullHistoryNaturalActionChunkR2RBatchBuilder,
    R2RStage3DataConfig,
    build_data_loader,
    stage2_data_config_from_dict,
)
from nav.v1.model import (  # noqa: E402
    SingleActionPolicyConfig,
    SingleActionStage3Model,
    build_model,
)


DEFAULT_STAGE2_CHECKPOINT = ROOT / (
    "log/v1_stage2_final_cotrain/"
    "stage2_from_step3000_continue_lr2e6_1k_20260825/"
    "checkpoints/step_003400.pt"
)
DEFAULT_R2R_DATA_CONFIG = R2RStage3DataConfig()


@dataclass(slots=True)
class TrainConfig:
    run_name: str
    checkpoint: str
    device: str
    steps: int
    batch_size: int
    grad_accum: int
    effective_batch_size: int
    r2r_loader: str
    action_chunk: int
    policy_loss: str
    backbone_lr: float
    policy_lr: float
    weight_decay: float
    grad_clip: float
    lambda_video_replay: float
    lambda_pose_replay: float
    dtype: str
    save_every: int
    log_every: int
    seed: int
    output_root: str
    tensorboard_port: int
    resume_optimizer: bool
    step_offset: int


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def install_non_reentrant_wan_checkpoint() -> None:
    try:
        import infworld.models.dit_model as dit_model_module
    except Exception:
        return

    def checkpoint_module(module, *args, **kwargs):
        if getattr(module, "grad_checkpointing", False):
            return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False, **kwargs)
        return module(*args, **kwargs)

    dit_model_module.auto_grad_checkpoint = checkpoint_module


def save_checkpoint(
    *,
    model: SingleActionStage3Model,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    step: int,
    train_cfg: TrainConfig,
    stage2_data_cfg: FinalStage2DataConfig,
    r2r_data_cfg: R2RStage3DataConfig,
) -> Path:
    path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.world_model.state_dict(),
            "single_action_policy_head": model.policy_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_config": asdict(train_cfg),
            "model_config": model.world_model.cfg.to_dict(),
            "single_action_policy_config": model.cfg.to_dict(),
            "stage2_replay_data_config": stage2_data_cfg.to_dict(),
            "r2r_data_config": r2r_data_cfg.to_dict(),
            "source_checkpoint": train_cfg.checkpoint,
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"stage3_r2r_single_action_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE2_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--policy-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-video-replay", type=float, default=0.25)
    parser.add_argument("--lambda-pose-replay", type=float, default=0.05)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output-root", type=Path, default=ROOT / "log/v1_stage3_r2r_single_action")
    parser.add_argument("--tensorboard-port", type=int, default=6039)
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="restore AdamW state from --checkpoint for an exact weight/optimizer continuation",
    )
    parser.add_argument(
        "--step-offset",
        type=int,
        default=0,
        help="absolute optimizer-step offset used for logging and checkpoint names",
    )
    parser.add_argument("--history-micro-choices", default="1,2,3,4,5,6,7")
    parser.add_argument("--action-chunk", type=int, default=1)
    parser.add_argument(
        "--policy-loss",
        choices=("cross_entropy", "inverse_frequency"),
        default="cross_entropy",
        help=(
            "policy objective; inverse_frequency keeps natural shuffled windows but weights each "
            "class by N/(C*N_c)"
        ),
    )
    parser.add_argument(
        "--r2r-loader",
        choices=(
            "stage3_r2r_balanced_single_action",
            "stage3_r2r_full_history_balanced_single_action",
            "stage3_r2r_full_history_natural_action_chunk",
        ),
        default="stage3_r2r_balanced_single_action",
    )
    parser.add_argument("--max-r2r-episodes", type=int, default=0)
    parser.add_argument("--latent-manifest-dir", type=Path, default=DEFAULT_R2R_DATA_CONFIG.latent_manifest_dir)
    parser.add_argument("--rendered-manifest", type=Path, default=DEFAULT_R2R_DATA_CONFIG.rendered_manifest)
    parser.add_argument("--r2r-text-cache-root", type=Path, default=DEFAULT_R2R_DATA_CONFIG.text_cache_root)
    parser.add_argument("--allow-empty-text-fallback", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size * args.grad_accum != 16:
        raise SystemExit("Stage3 requires effective batch size = 16")
    if not 1 <= args.action_chunk <= 10:
        raise SystemExit("action chunk must be in [1, 10], matching the shared action-token capacity")
    balanced_loader = args.r2r_loader in {
        "stage3_r2r_balanced_single_action",
        "stage3_r2r_full_history_balanced_single_action",
    }
    if balanced_loader and (args.batch_size * args.grad_accum) % 4 != 0:
        raise SystemExit("effective batch size must be divisible by 4 for exact per-step class balance")
    if balanced_loader and args.action_chunk != 1:
        raise SystemExit("historical balanced loaders only support action chunk = 1")
    if args.r2r_loader == "stage3_r2r_full_history_natural_action_chunk" and args.action_chunk != 4:
        raise SystemExit("formal full-history natural loader requires action chunk = 4")
    os.environ.setdefault("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_non_reentrant_wan_checkpoint()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    policy_cfg_override = SingleActionPolicyConfig(
        hidden_dim=1536,
        policy_queries=args.action_chunk,
        backbone_action_tokens=(1 if balanced_loader else 10),
        sampler=("balanced_class_cycle" if balanced_loader else "natural_window_no_balance"),
        loss_type=args.policy_loss,
    )
    model_assembly = build_model(
        "stage3_single_action_checkpoint",
        checkpoint=args.checkpoint,
        policy_config=policy_cfg_override,
        device=device,
        dtype=dtype,
        training=True,
    )
    model = model_assembly.model
    if not isinstance(model, SingleActionStage3Model):
        raise TypeError(type(model))
    world_model = model.world_model
    source_ckpt = model_assembly.source_payload
    if source_ckpt is None:
        raise RuntimeError("Stage3 assembly did not return its Stage2 source payload")
    if args.resume_optimizer:
        if "optimizer" not in source_ckpt:
            raise RuntimeError("--resume-optimizer requested but checkpoint has no optimizer state")
        source_step = int(source_ckpt.get("step", -1))
        if args.step_offset != source_step:
            raise RuntimeError(
                f"resume step mismatch: --step-offset={args.step_offset}, checkpoint step={source_step}"
            )
    elif args.step_offset:
        raise RuntimeError("--step-offset requires --resume-optimizer")
    policy_cfg = model.cfg

    source_data_payload = source_ckpt.get("data_config") or source_ckpt.get("stage2_replay_data_config")
    if source_data_payload is None:
        raise RuntimeError("source Stage2 checkpoint has no data_config")
    stage2_data_cfg = stage2_data_config_from_dict(source_data_payload)
    stage2_data_cfg.batch_size = args.batch_size
    stage2_data_cfg.seed = args.seed + 1000
    replay_builder = build_data_loader("stage2_mixed_video", config=stage2_data_cfg)
    if not isinstance(replay_builder, FinalStage2BatchBuilder):
        raise TypeError(type(replay_builder))
    r2r_data_cfg = R2RStage3DataConfig(
        latent_manifest_dir=args.latent_manifest_dir,
        rendered_manifest=args.rendered_manifest,
        text_empty=stage2_data_cfg.text_empty,
        text_cache_root=args.r2r_text_cache_root,
        history_micro_choices=args.history_micro_choices,
        action_horizon=args.action_chunk,
        batch_size=args.batch_size,
        require_text_cache=not args.allow_empty_text_fallback,
        max_episodes=args.max_r2r_episodes,
        action_oversample_mode="none",
        seed=args.seed,
    )
    r2r_builder = build_data_loader(
        args.r2r_loader,
        config=r2r_data_cfg,
        history_action_horizon=policy_cfg.history_action_horizon,
        model_action_horizon=policy_cfg.backbone_action_tokens,
    )
    if not isinstance(
        r2r_builder,
        (
            BalancedSingleActionR2RBatchBuilder,
            FullHistoryBalancedSingleActionR2RBatchBuilder,
            FullHistoryNaturalActionChunkR2RBatchBuilder,
        ),
    ):
        raise TypeError(type(r2r_builder))
    r2r_summary = r2r_builder.summary()
    if args.policy_loss == "inverse_frequency":
        if not isinstance(r2r_builder, FullHistoryNaturalActionChunkR2RBatchBuilder):
            raise SystemExit("inverse-frequency loss currently requires the natural H=4 loader")
        label_counts = r2r_summary.get("action_label_counts")
        if not isinstance(label_counts, dict):
            raise RuntimeError("natural H=4 loader did not report action_label_counts")
        model.configure_inverse_frequency_class_weights(
            [int(label_counts[ACTION_NAMES[action_id]]) for action_id in sorted(ACTION_NAMES)]
        )

    train_cfg = TrainConfig(
        run_name=args.run_name,
        checkpoint=str(args.checkpoint),
        device=str(device),
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        effective_batch_size=args.batch_size * args.grad_accum,
        r2r_loader=args.r2r_loader,
        action_chunk=args.action_chunk,
        policy_loss=args.policy_loss,
        backbone_lr=args.backbone_lr,
        policy_lr=args.policy_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_video_replay=args.lambda_video_replay,
        lambda_pose_replay=args.lambda_pose_replay,
        dtype=args.dtype,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        output_root=str(args.output_root),
        tensorboard_port=args.tensorboard_port,
        resume_optimizer=args.resume_optimizer,
        step_offset=args.step_offset,
    )
    fast_parameters = model.policy_fast_parameters()
    backbone_parameters = model.backbone_parameters()
    full_history_mode = isinstance(
        r2r_builder,
        (FullHistoryBalancedSingleActionR2RBatchBuilder, FullHistoryNaturalActionChunkR2RBatchBuilder),
    )
    natural_action_chunk_mode = isinstance(r2r_builder, FullHistoryNaturalActionChunkR2RBatchBuilder)
    preflight = {
        "event": "stage3_r2r_single_action_preflight",
        "time": now(),
        "status": (
            "full_model_stage3_full_episode_prefix_ablation"
            if full_history_mode
            else "diagnostic_full_model_single_action; temporal representation unchanged"
        ),
        "train": asdict(train_cfg),
        "r2r": r2r_summary,
        "replay_data": replay_builder.summary(),
        "model": model.structural_report(),
        "source_checkpoint_step": int(source_ckpt["step"]),
        "optimizer_groups": {
            "backbone_register_video_pose": {
                "parameters": sum(parameter.numel() for parameter in backbone_parameters),
                "lr": args.backbone_lr,
                "dtype": args.dtype,
            },
            "policy_head_and_action_query_input": {
                "parameters": sum(parameter.numel() for parameter in fast_parameters),
                "lr": args.policy_lr,
                "dtype": "fp32",
            },
        },
        "sample_rule": {
            "history": r2r_builder.summary().get(
                "history_semantics",
                "existing T4 K=1..7 sampled Register rollout",
            ),
            "z_obs": "existing current T4 micro chunk; temporal alignment intentionally unchanged",
            "target": (
                f"{args.action_chunk} consecutive future primitives starting at "
                "label_start=obs_micro*12+12"
            ),
            "balance": (
                (
                    "natural shuffled window traversal; no sample duplication; "
                    "inverse-frequency class-balanced CE"
                    if args.policy_loss == "inverse_frequency"
                    else "natural shuffled window traversal; no class balancing or sample duplication"
                )
                if natural_action_chunk_mode
                else "logical copy via exact STOP/MOVE/LEFT/RIGHT cyclic sampling"
            ),
        },
        "loss": {
            "policy": model.structural_report()["policy_loss"],
            "total": (
                f"L = L_policy(H={args.action_chunk}) + lambda_video_replay * L_visual + "
                "lambda_pose_replay * L_pose"
            ),
        },
    }
    write_json(run_dir / "formal_preflight.json", preflight)
    write_json(
        run_dir / "config.json",
        {
            "train": asdict(train_cfg),
            "r2r_data": r2r_data_cfg.to_dict(),
            "stage2_replay_data": stage2_data_cfg.to_dict(),
            "model": world_model.cfg.to_dict(),
            "single_action_policy": policy_cfg.to_dict(),
        },
    )
    if args.preflight_only:
        print(json.dumps({"status": "preflight_ok", "run_dir": str(run_dir), "r2r": preflight["r2r"]}, indent=2, ensure_ascii=False))
        return

    try:
        from infworld.models.checkpoint import set_grad_checkpoint

        set_grad_checkpoint(model.world_model.backbone)
    except Exception:
        pass
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr, "name": "backbone"},
            {"params": fast_parameters, "lr": args.policy_lr, "name": "policy_fp32"},
        ],
        weight_decay=args.weight_decay,
    )
    if args.resume_optimizer:
        optimizer.load_state_dict(source_ckpt["optimizer"])
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    log_path = run_dir / "train.jsonl"
    sampled_target_counts: Counter[int] = Counter()
    sampled_prediction_counts: Counter[int] = Counter()
    with log_path.open("a") as log:
        log.write(json.dumps({"event": "start", "time": now(), "run_dir": str(run_dir), **preflight}, ensure_ascii=False) + "\n")
        last = time.time()
        global_start = time.time()
        for local_step in range(1, args.steps + 1):
            step = args.step_offset + local_step
            optimizer.zero_grad(set_to_none=True)
            metrics: dict[str, float] = {}
            step_target_counts: Counter[int] = Counter()
            step_correct_counts: Counter[int] = Counter()
            step_prediction_counts: Counter[int] = Counter()
            r2r_meta_last: dict[str, Any] = {}
            replay_meta_last: dict[str, Any] = {}
            for _ in range(args.grad_accum):
                r2r_batch, r2r_meta = r2r_builder.next_batch(device=device, dtype=dtype)
                out = model.forward_policy(r2r_batch)
                policy_loss = out["loss"] / args.grad_accum
                if not torch.isfinite(policy_loss):
                    raise FloatingPointError(f"non-finite policy loss at step {step}: {policy_loss}")
                policy_loss.backward()
                metrics["loss_policy"] = metrics.get("loss_policy", 0.0) + float(out["loss"].detach().cpu()) / args.grad_accum
                metrics["loss_policy_unweighted_ce"] = (
                    metrics.get("loss_policy_unweighted_ce", 0.0)
                    + float(out["loss_ce"].detach().cpu()) / args.grad_accum
                )
                target = r2r_batch["action_class"].detach().cpu().reshape(-1)
                prediction = out["action_pred"].detach().cpu().reshape(-1)
                for target_id, pred_id in zip(target.tolist(), prediction.tolist()):
                    target_id = int(target_id)
                    pred_id = int(pred_id)
                    step_target_counts[target_id] += 1
                    step_prediction_counts[pred_id] += 1
                    sampled_target_counts[target_id] += 1
                    sampled_prediction_counts[pred_id] += 1
                    step_correct_counts[target_id] += int(target_id == pred_id)
                r2r_meta_last = r2r_meta

                replay_batch, replay_meta = replay_builder.next_batch(device=device, dtype=dtype)
                replay_out = model.world_model.forward_stage2(replay_batch, lambda_pose=args.lambda_pose_replay)
                replay_loss = (
                    float(args.lambda_video_replay) * replay_out["loss_visual_tensor"]
                    + float(args.lambda_pose_replay) * replay_out["loss_pose_tensor"]
                ) / args.grad_accum
                if not torch.isfinite(replay_loss):
                    raise FloatingPointError(f"non-finite replay loss at step {step}: {replay_loss}")
                replay_loss.backward()
                metrics["loss_video_replay"] = metrics.get("loss_video_replay", 0.0) + float(replay_out["loss_visual"].detach().float().cpu()) / args.grad_accum
                metrics["loss_pose_replay"] = metrics.get("loss_pose_replay", 0.0) + float(replay_out["loss_pose"].detach().float().cpu()) / args.grad_accum
                replay_meta_last = replay_meta

            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                metrics["grad_norm"] = float(grad_norm.detach().float().cpu())
            optimizer.step()
            model.keep_policy_modules_fp32()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step % args.log_every == 0 or step == 1:
                now_time = time.time()
                correct = sum(step_correct_counts.values())
                valid = sum(step_target_counts.values())
                record: dict[str, Any] = {
                    "event": "train",
                    "time": now(),
                    "step": step,
                    "seconds_per_step_window": (now_time - last) / max(args.log_every, 1),
                    "seconds_total": now_time - global_start,
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    "lr_policy": optimizer.param_groups[1]["lr"],
                    "r2r_history_micro": r2r_meta_last.get("history_micro"),
                    "r2r_history_micro_min": r2r_meta_last.get("history_micro_min"),
                    "r2r_history_micro_max": r2r_meta_last.get("history_micro_max"),
                    "r2r_history_micro_mean": r2r_meta_last.get("history_micro_mean"),
                    "r2r_text_hits": r2r_meta_last.get("text_hits"),
                    "r2r_text_fallbacks": r2r_meta_last.get("text_fallbacks"),
                    "replay_history_iw": replay_meta_last.get("history_iw"),
                    "replay_datasets": replay_meta_last.get("datasets"),
                    "train/action_accuracy": float(correct) / float(max(valid, 1)),
                    **{f"train/{key}": value for key, value in metrics.items()},
                }
                recalls = []
                for action_id, action_name in ACTION_NAMES.items():
                    count = step_target_counts[action_id]
                    recall = float(step_correct_counts[action_id]) / float(max(count, 1))
                    recalls.append(recall)
                    record[f"train/target_count_{action_name}"] = count
                    record[f"train/pred_count_{action_name}"] = step_prediction_counts[action_id]
                    record[f"train/recall_{action_name}"] = recall
                record["train/macro_recall"] = sum(recalls) / len(recalls)
                if device.type == "cuda":
                    record["cuda_max_memory_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                last = now_time
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, step)
            if step % args.save_every == 0 or local_step == args.steps:
                checkpoint = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    run_dir=run_dir,
                    step=step,
                    train_cfg=train_cfg,
                    stage2_data_cfg=stage2_data_cfg,
                    r2r_data_cfg=r2r_data_cfg,
                )
                log.write(json.dumps({"event": "checkpoint", "time": now(), "step": step, "path": str(checkpoint)}, ensure_ascii=False) + "\n")
                log.flush()
    writer.close()
    print(json.dumps({"status": "ok", "run_dir": str(run_dir), "log": str(log_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
