#!/usr/bin/env python3
"""Final Stage3 R2R policy training with future-action chunks.

This entrypoint uses the full NAV V1 model path.  The R2R dataloader follows
the current Stage3 rule: one sample predicts exactly one future action chunk
after ``Z_obs``; repeated terminal padding chunks are not sampled as repeated
``STOP`` targets.
"""

from __future__ import annotations

import argparse
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

from nav.v1.stage2_final import FinalStage2BatchBuilder, FinalStage2DataConfig, FinalStage2WanConfig, FinalStage2WanModel  # noqa: E402
from nav.v1.stage3_r2r import R2RStage3DataConfig, R2RStage3PolicyBatchBuilder  # noqa: E402


DEFAULT_STAGE2_CHECKPOINT = ROOT / "log/v1_stage2_final_cotrain/stage2_final_from_wan_probe_l16_fullmix_2k_20260822_001/checkpoints/step_001600.pt"
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
    lr: float
    weight_decay: float
    grad_clip: float
    lambda_ce: float
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


def load_stage2_model(checkpoint: Path, *, device: torch.device, dtype: torch.dtype) -> tuple[FinalStage2WanModel, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = make_dataclass(FinalStage2WanConfig, ckpt["model_config"])
    model = FinalStage2WanModel(cfg)
    model.remove_hmpc()
    result = model.load_state_dict(ckpt["model"], strict=False)
    bad_missing = [key for key in result.missing_keys if not key.startswith("backbone.latent_encoder.")]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing[:20]} unexpected={result.unexpected_keys[:20]}")
    return model.to(device=device, dtype=dtype).train(), ckpt


def save_checkpoint(
    *,
    model: FinalStage2WanModel,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    step: int,
    train_cfg: TrainConfig,
    model_cfg: FinalStage2WanConfig,
    stage2_data_cfg: FinalStage2DataConfig,
    r2r_data_cfg: R2RStage3DataConfig,
) -> Path:
    path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_config": asdict(train_cfg),
            "model_config": model_cfg.to_dict(),
            "stage2_replay_data_config": stage2_data_cfg.to_dict(),
            "r2r_data_config": r2r_data_cfg.to_dict(),
            "source_checkpoint": train_cfg.checkpoint,
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"stage3_r2r_future_action_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE2_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-video-replay", type=float, default=0.25)
    parser.add_argument("--lambda-pose-replay", type=float, default=0.05)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output-root", type=Path, default=ROOT / "log" / "v1_stage3_r2r_future_action")
    parser.add_argument("--tensorboard-port", type=int, default=6036)
    parser.add_argument("--history-micro-choices", default="1,2,3,4,5,6,7")
    parser.add_argument("--max-r2r-episodes", type=int, default=0)
    parser.add_argument("--latent-manifest-dir", type=Path, default=DEFAULT_R2R_DATA_CONFIG.latent_manifest_dir)
    parser.add_argument("--rendered-manifest", type=Path, default=DEFAULT_R2R_DATA_CONFIG.rendered_manifest)
    parser.add_argument("--r2r-text-cache-root", type=Path, default=DEFAULT_R2R_DATA_CONFIG.text_cache_root)
    parser.add_argument("--allow-empty-text-fallback", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size * args.grad_accum != 16:
        raise SystemExit("final Stage3 R2R training requires effective batch size = 16")
    os.environ.setdefault("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_non_reentrant_wan_checkpoint()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    model, source_ckpt = load_stage2_model(args.checkpoint, device=device, dtype=dtype)
    stage2_data_cfg = make_dataclass(FinalStage2DataConfig, source_ckpt["data_config"])
    stage2_data_cfg.batch_size = args.batch_size
    stage2_data_cfg.seed = args.seed + 1000
    replay_builder = FinalStage2BatchBuilder(stage2_data_cfg)
    r2r_data_cfg = R2RStage3DataConfig(
        latent_manifest_dir=args.latent_manifest_dir,
        rendered_manifest=args.rendered_manifest,
        text_empty=stage2_data_cfg.text_empty,
        text_cache_root=args.r2r_text_cache_root,
        history_micro_choices=args.history_micro_choices,
        action_horizon=model.cfg.action_horizon,
        batch_size=args.batch_size,
        require_text_cache=not args.allow_empty_text_fallback,
        max_episodes=args.max_r2r_episodes,
        seed=args.seed,
    )
    r2r_builder = R2RStage3PolicyBatchBuilder(r2r_data_cfg)

    train_cfg = TrainConfig(
        run_name=args.run_name,
        checkpoint=str(args.checkpoint),
        device=str(device),
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        effective_batch_size=args.batch_size * args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_ce=args.lambda_ce,
        lambda_video_replay=args.lambda_video_replay,
        lambda_pose_replay=args.lambda_pose_replay,
        dtype=args.dtype,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        output_root=str(args.output_root),
        tensorboard_port=args.tensorboard_port,
    )
    preflight = {
        "event": "stage3_r2r_future_action_preflight",
        "time": now(),
        "binding_standard": "doc/00_overview/final_formal_training_standard.md",
        "train": asdict(train_cfg),
        "r2r": r2r_builder.summary(),
        "replay_data": replay_builder.summary(),
        "model": model.structural_report(),
        "source_checkpoint_step": int(source_ckpt["step"]),
        "sample_rule": {
            "history": "K previous T4 micro chunks update Register with A_hist",
            "z_obs": "current observed T4 micro chunk",
            "target_action": "one future action chunk after z_obs; label_start=obs_micro*12+12",
            "terminal": "one terminal STOP action chunk per episode at most; repeated terminal visual padding is not sampled",
        },
        "loss": "L = lambda_ce * CE(combo_logits, action_combo) + lambda_video_replay * L_visual_replay + lambda_pose_replay * L_pose_replay",
    }
    write_json(run_dir / "formal_preflight.json", preflight)
    write_json(
        run_dir / "config.json",
        {
            "train": asdict(train_cfg),
            "r2r_data": r2r_data_cfg.to_dict(),
            "stage2_replay_data": stage2_data_cfg.to_dict(),
            "model": model.cfg.to_dict(),
        },
    )
    if args.preflight_only:
        print(json.dumps({"status": "preflight_ok", "run_dir": str(run_dir), "r2r": preflight["r2r"]}, indent=2, ensure_ascii=False))
        return

    try:
        from infworld.models.checkpoint import set_grad_checkpoint

        set_grad_checkpoint(model.backbone)
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    log_path = run_dir / "train.jsonl"
    with log_path.open("a") as log:
        log.write(json.dumps({"event": "start", "time": now(), "run_dir": str(run_dir), **preflight}, ensure_ascii=False) + "\n")
        last = time.time()
        global_start = time.time()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            metrics: dict[str, float] = {}
            r2r_meta_last: dict[str, Any] = {}
            replay_meta_last: dict[str, Any] = {}
            for _ in range(args.grad_accum):
                r2r_batch, r2r_meta = r2r_builder.next_batch(device=device, dtype=dtype)
                out = model.forward_stage3_policy(r2r_batch, lambda_ce=args.lambda_ce)
                loss = out["loss"] / args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite policy loss at step {step}: {loss}")
                loss.backward()
                metrics["loss_policy"] = metrics.get("loss_policy", 0.0) + float(out["loss"].detach().float().cpu().item()) / args.grad_accum
                metrics["loss_ce_aux"] = metrics.get("loss_ce_aux", 0.0) + float(out["loss_ce_aux"].detach().float().cpu().item()) / args.grad_accum
                metrics["loss_action_flow"] = metrics.get("loss_action_flow", 0.0) + float(out["loss_action_flow"].detach().float().cpu().item()) / args.grad_accum
                r2r_meta_last = r2r_meta

                replay_batch, replay_meta = replay_builder.next_batch(device=device, dtype=dtype)
                replay_out = model.forward_stage2(replay_batch, lambda_pose=args.lambda_pose_replay)
                replay_loss = (
                    float(args.lambda_video_replay) * replay_out["loss_visual_tensor"]
                    + float(args.lambda_pose_replay) * replay_out["loss_pose_tensor"]
                ) / args.grad_accum
                if not torch.isfinite(replay_loss):
                    raise FloatingPointError(f"non-finite replay loss at step {step}: {replay_loss}")
                replay_loss.backward()
                metrics["loss_video_replay"] = metrics.get("loss_video_replay", 0.0) + float(replay_out["loss_visual"].detach().float().cpu().item()) / args.grad_accum
                metrics["loss_pose_replay"] = metrics.get("loss_pose_replay", 0.0) + float(replay_out["loss_pose"].detach().float().cpu().item()) / args.grad_accum
                replay_meta_last = replay_meta

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step % args.log_every == 0 or step == 1:
                now_time = time.time()
                record = {
                    "event": "train",
                    "time": now(),
                    "step": step,
                    "seconds_per_step_window": (now_time - last) / max(args.log_every, 1),
                    "seconds_total": now_time - global_start,
                    "lr": optimizer.param_groups[0]["lr"],
                    "r2r_history_micro": r2r_meta_last.get("history_micro"),
                    "r2r_terminal_count": r2r_meta_last.get("terminal_count"),
                    "r2r_action_valid": r2r_meta_last.get("action_valid"),
                    "r2r_text_hits": r2r_meta_last.get("text_hits"),
                    "r2r_text_fallbacks": r2r_meta_last.get("text_fallbacks"),
                    "replay_history_iw": replay_meta_last.get("history_iw"),
                    "replay_datasets": replay_meta_last.get("datasets"),
                    **{f"train/{key}": value for key, value in metrics.items()},
                }
                if device.type == "cuda":
                    record["cuda_max_memory_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                last = now_time
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, step)
            if step % args.save_every == 0 or step == args.steps:
                ckpt = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    run_dir=run_dir,
                    step=step,
                    train_cfg=train_cfg,
                    model_cfg=model.cfg,
                    stage2_data_cfg=stage2_data_cfg,
                    r2r_data_cfg=r2r_data_cfg,
                )
                log.write(json.dumps({"event": "checkpoint", "time": now(), "step": step, "path": str(ckpt)}, ensure_ascii=False) + "\n")
                log.flush()
    writer.close()
    print(json.dumps({"status": "ok", "run_dir": str(run_dir), "log": str(log_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
