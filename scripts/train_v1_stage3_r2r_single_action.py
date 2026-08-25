#!/usr/bin/env python3
"""Train the full V1 model on a balanced one-step R2R policy target.

The video/Register/pose path is the complete current Stage2 implementation.
The diagnostic change is deliberately narrow: one deterministic action query
is read from the final shared Wan hidden and decoded by an FP32 four-class MLP.
R2R minority classes are sampled with replacement in an exact four-class
cycle, while Stage2 visual/pose replay remains enabled.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, fields
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

from nav.v1.stage2_final import (  # noqa: E402
    FinalStage2BatchBuilder,
    FinalStage2DataConfig,
    FinalStage2WanConfig,
    FinalStage2WanModel,
)
from nav.v1.stage3_r2r import ACTION_NAMES, R2RStage3DataConfig  # noqa: E402
from nav.v1.stage3_single_action import (  # noqa: E402
    BalancedSingleActionR2RBatchBuilder,
    SingleActionPolicyConfig,
    SingleActionStage3Model,
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


def make_dataclass(cls, payload: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    clean = {key: value for key, value in payload.items() if key in allowed}
    for key in (
        "manifest",
        "latent_root",
        "text_empty",
        "text_cache_root",
        "re10k_camera_root",
        "latent_manifest_dir",
        "rendered_manifest",
    ):
        if key in clean:
            clean[key] = Path(clean[key])
    if cls is FinalStage2WanConfig and "register_condition_grid" in clean:
        clean["register_condition_grid"] = tuple(clean["register_condition_grid"])
    return cls(**clean)


def load_stage2_world(
    checkpoint: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[FinalStage2WanModel, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = make_dataclass(FinalStage2WanConfig, payload["model_config"])
    model = FinalStage2WanModel(cfg)
    model.remove_hmpc()
    result = model.load_state_dict(payload["model"], strict=False)
    bad_missing = [key for key in result.missing_keys if not key.startswith("backbone.latent_encoder.")]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Stage2 checkpoint mismatch: missing={bad_missing[:20]} unexpected={result.unexpected_keys[:20]}"
        )
    model._freeze_unused_action_output_heads()
    return model.to(device=device, dtype=dtype).train(), payload


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
    parser.add_argument("--history-micro-choices", default="1,2,3,4,5,6,7")
    parser.add_argument("--max-r2r-episodes", type=int, default=0)
    parser.add_argument("--latent-manifest-dir", type=Path, default=DEFAULT_R2R_DATA_CONFIG.latent_manifest_dir)
    parser.add_argument("--rendered-manifest", type=Path, default=DEFAULT_R2R_DATA_CONFIG.rendered_manifest)
    parser.add_argument("--r2r-text-cache-root", type=Path, default=DEFAULT_R2R_DATA_CONFIG.text_cache_root)
    parser.add_argument("--allow-empty-text-fallback", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size != 1:
        raise SystemExit("balanced single-action Stage3 currently requires physical batch size = 1")
    if args.batch_size * args.grad_accum != 16:
        raise SystemExit("single-action Stage3 requires effective batch size = 16")
    if args.grad_accum % 4 != 0:
        raise SystemExit("grad_accum must be divisible by 4 for exact per-step class balance")
    os.environ.setdefault("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_non_reentrant_wan_checkpoint()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    world_model, source_ckpt = load_stage2_world(args.checkpoint, device=device, dtype=dtype)
    policy_cfg = SingleActionPolicyConfig(hidden_dim=world_model.cfg.hidden_dim)
    model = SingleActionStage3Model(world_model, policy_cfg).to(device=device)
    model.keep_policy_modules_fp32()

    source_data_payload = source_ckpt.get("data_config") or source_ckpt.get("stage2_replay_data_config")
    if source_data_payload is None:
        raise RuntimeError("source Stage2 checkpoint has no data_config")
    stage2_data_cfg = make_dataclass(FinalStage2DataConfig, source_data_payload)
    stage2_data_cfg.batch_size = args.batch_size
    stage2_data_cfg.seed = args.seed + 1000
    replay_builder = FinalStage2BatchBuilder(stage2_data_cfg)
    r2r_data_cfg = R2RStage3DataConfig(
        latent_manifest_dir=args.latent_manifest_dir,
        rendered_manifest=args.rendered_manifest,
        text_empty=stage2_data_cfg.text_empty,
        text_cache_root=args.r2r_text_cache_root,
        history_micro_choices=args.history_micro_choices,
        action_horizon=1,
        batch_size=args.batch_size,
        require_text_cache=not args.allow_empty_text_fallback,
        max_episodes=args.max_r2r_episodes,
        action_oversample_mode="none",
        seed=args.seed,
    )
    r2r_builder = BalancedSingleActionR2RBatchBuilder(
        r2r_data_cfg,
        history_action_horizon=policy_cfg.history_action_horizon,
    )

    train_cfg = TrainConfig(
        run_name=args.run_name,
        checkpoint=str(args.checkpoint),
        device=str(device),
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        effective_batch_size=args.batch_size * args.grad_accum,
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
    )
    fast_parameters = model.policy_fast_parameters()
    backbone_parameters = model.backbone_parameters()
    preflight = {
        "event": "stage3_r2r_single_action_preflight",
        "time": now(),
        "status": "diagnostic_full_model_single_action; temporal representation unchanged",
        "train": asdict(train_cfg),
        "r2r": r2r_builder.summary(),
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
            "history": "existing T4 K=1..7 Register rollout; unchanged for this diagnostic",
            "z_obs": "existing current T4 micro chunk; temporal alignment intentionally unchanged",
            "target": "one primitive at label_start=obs_micro*12+12",
            "balance": "logical copy via exact STOP/MOVE/LEFT/RIGHT cyclic sampling",
        },
        "loss": "L = CE_4(next_action) + lambda_video_replay * L_visual + lambda_pose_replay * L_pose",
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
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    log_path = run_dir / "train.jsonl"
    sampled_target_counts: Counter[int] = Counter()
    sampled_prediction_counts: Counter[int] = Counter()
    with log_path.open("a") as log:
        log.write(json.dumps({"event": "start", "time": now(), "run_dir": str(run_dir), **preflight}, ensure_ascii=False) + "\n")
        last = time.time()
        global_start = time.time()
        for step in range(1, args.steps + 1):
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
            if step % args.save_every == 0 or step == args.steps:
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
