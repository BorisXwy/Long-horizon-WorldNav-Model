#!/usr/bin/env python3
"""Final-standard V1 Stage2 cotrain entrypoint.

This is intentionally separate from older RE10K/IW-aligned scripts so runs can
be compared without ambiguity.  It uses the real Wan/IW DiT implementation,
T_latent=4 data, mixed RE10K/SpatialVID/DL3DV windows, Z_obs in the main token
stream, A_cur as condition, A_noise as main action tokens, and visual+pose loss.
"""

from __future__ import annotations

import argparse
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

from nav.v1.stage2_final import (  # noqa: E402
    FinalStage2BatchBuilder,
    FinalStage2DataConfig,
    FinalStage2WanConfig,
    FinalStage2WanModel,
)


WAN_DEFAULT = Path("/sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
DEFAULT_MANIFEST = Path("/sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes_spatial20.jsonl")
DEFAULT_LATENT_ROOT = Path("/sharedata/NAV/derived/v1/t4_micro_latents_spatial20")
DEFAULT_DATASET_WEIGHTS = "re10k=0.20,spatialvid=0.45,dl3dv=0.35"
DEFAULT_HISTORY_IW = "1,4,8,16"


@dataclass(slots=True)
class TrainConfig:
    run_name: str
    device: str
    steps: int
    batch_size: int
    grad_accum: int
    effective_batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    lambda_pose: float
    dtype: str
    pose_head_type: str
    pose_readout_layer: int
    save_every: int
    log_every: int
    seed: int
    checkpoint: str
    resume_checkpoint: str | None
    resume_from_step: int
    resume_optimizer: bool
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
    """Match the stable Stage1 Wan checkpointing behavior.

    The patched WanBlock keeps hist/prefix caches during forward.  PyTorch's
    default reentrant checkpoint can retain references that fail across
    gradient-accumulation micro-batches.  The earlier stable V1 Stage1 script
    used the same non-reentrant override.
    """

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
    model: FinalStage2WanModel,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    step: int,
    train_cfg: TrainConfig,
    model_cfg: FinalStage2WanConfig,
    data_cfg: FinalStage2DataConfig,
    wan_audit: dict[str, Any],
) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_config": asdict(train_cfg),
            "model_config": model_cfg.to_dict(),
            "data_config": data_cfg.to_dict(),
            "wan_init_audit": wan_audit,
        },
        path,
    )
    return path


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move restored optimizer buffers onto the active training device."""

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_training_checkpoint(
    *,
    model: FinalStage2WanModel,
    optimizer: torch.optim.Optimizer | None,
    checkpoint: Path,
    device: torch.device,
    resume_optimizer: bool,
) -> dict[str, Any]:
    """Resume an internal NAV checkpoint after the model graph is constructed.

    `--checkpoint` remains the immutable Wan / base init.  This function is for
    NAV checkpoints saved by `save_checkpoint`, which contain the full V1 model
    state plus optimizer state.  Keeping the two paths separate prevents an
    accidental "fake resume" where a NAV checkpoint is interpreted as raw Wan
    weights.
    """

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"resume checkpoint must be a dict payload: {checkpoint}")
    model_state = payload.get("model")
    if model_state is None:
        raise RuntimeError(f"resume checkpoint has no full model state under key 'model': {checkpoint}")
    model.load_state_dict(model_state, strict=True)
    optimizer_loaded = False
    if optimizer is not None and resume_optimizer:
        opt_state = payload.get("optimizer")
        if opt_state is None:
            raise RuntimeError(f"resume optimizer requested but checkpoint has no optimizer state: {checkpoint}")
        optimizer.load_state_dict(opt_state)
        move_optimizer_state_to_device(optimizer, device)
        optimizer_loaded = True
    return {
        "checkpoint": str(checkpoint),
        "step": int(payload.get("step", 0)),
        "has_optimizer": payload.get("optimizer") is not None,
        "optimizer_loaded": optimizer_loaded,
        "train_config": payload.get("train_config"),
        "model_config": payload.get("model_config"),
        "data_config": payload.get("data_config"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=f"stage2_final_cotrain_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-pose", type=float, default=0.1)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--pose-head-type", choices=("frame_mlp", "dense_camera_query"), default="frame_mlp")
    parser.add_argument("--pose-readout-layer", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--checkpoint", type=Path, default=WAN_DEFAULT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--no-resume-optimizer", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "log" / "v1_stage2_final_cotrain")
    parser.add_argument("--tensorboard-port", type=int, default=6017)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--dataset-weights", default=DEFAULT_DATASET_WEIGHTS)
    parser.add_argument("--history-iw-chunks", default=DEFAULT_HISTORY_IW)
    parser.add_argument("--empty-hist-prob", type=float, default=0.10)
    parser.add_argument("--empty-cur-prob", type=float, default=0.10)
    parser.add_argument("--register-injection", choices=("main_prefix", "condition_memory"), default="condition_memory")
    parser.add_argument("--register-grad-tail", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size * args.grad_accum != 16:
        raise SystemExit("final Stage2 cotrain requires effective batch size = 16")
    os.environ.setdefault("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_non_reentrant_wan_checkpoint()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    train_cfg = TrainConfig(
        run_name=args.run_name,
        device=str(device),
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        effective_batch_size=args.batch_size * args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_pose=args.lambda_pose,
        dtype=args.dtype,
        pose_head_type=args.pose_head_type,
        pose_readout_layer=args.pose_readout_layer,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        checkpoint=str(args.checkpoint),
        resume_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None,
        resume_from_step=0,
        resume_optimizer=not args.no_resume_optimizer,
        output_root=str(args.output_root),
        tensorboard_port=args.tensorboard_port,
    )
    data_cfg = FinalStage2DataConfig(
        manifest=args.manifest,
        latent_root=args.latent_root,
        dataset_weights=args.dataset_weights,
        history_iw_chunks=args.history_iw_chunks,
        batch_size=args.batch_size,
        empty_hist_prob=args.empty_hist_prob,
        empty_cur_prob=args.empty_cur_prob,
        seed=args.seed,
    )
    model_cfg = FinalStage2WanConfig(
        register_injection=args.register_injection,
        register_grad_tail=args.register_grad_tail,
        pose_head_type=args.pose_head_type,
        pose_readout_layer=args.pose_readout_layer,
    )
    builder = FinalStage2BatchBuilder(data_cfg)
    model = FinalStage2WanModel(model_cfg)
    wan_audit = model.load_wan_checkpoint(args.checkpoint)
    preflight = {
        "event": "stage2_final_cotrain_preflight",
        "time": now(),
        "binding_standard": "doc/00_overview/final_formal_training_standard.md",
        "train_config": asdict(train_cfg),
        "data": builder.summary(),
        "model": model.structural_report(),
        "wan_init_audit": {
            **wan_audit,
            "missing_keys_count": len(wan_audit.get("missing_keys", [])),
        },
        "input_boundary": {
            "main_token_stream": (
                ["Register R_t", "Z_obs", "A_noise", "Z_future_noise"]
                if args.register_injection == "main_prefix"
                else ["Z_obs", "A_noise", "Z_future_noise"]
            ),
            "condition": (
                ["text", "A_cur"]
                if args.register_injection == "main_prefix"
                else ["Register R_t", "text", "A_cur"]
            ),
            "pre_backbone_rollout": ["C_hist", "A_hist"],
            "target_loss_only": ["Z_future_target", "pose_target/G_3D"],
        },
        "register_bptt": {
            "mode": "full_history_state_truncated_gradient",
            "register_grad_tail": args.register_grad_tail,
            "description": "C_hist 全部用于滚动 Register；为支持 IW16 长历史，只有最后 register_grad_tail 个 micro history update 保留反向图。",
        },
        "loss": "L_stage2 = L_visual + lambda_pose * L_pose; pose_mask=0 samples contribute only visual",
    }
    write_json(run_dir / "formal_preflight.json", preflight)
    write_json(run_dir / "config.json", {"train": asdict(train_cfg), "data": data_cfg.to_dict(), "model": model_cfg.to_dict()})
    write_json(run_dir / "wan_init_audit.json", wan_audit)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_ok", "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
        return

    model.to(device=device, dtype=torch_dtype(args.dtype)).train()
    try:
        from infworld.models.checkpoint import set_grad_checkpoint

        set_grad_checkpoint(model.backbone)
    except Exception:
        pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    log_path = run_dir / "train.jsonl"
    resume_audit: dict[str, Any] | None = None
    resume_from_step = 0
    if args.resume_checkpoint is not None:
        resume_audit = load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint=args.resume_checkpoint,
            device=device,
            resume_optimizer=not args.no_resume_optimizer,
        )
        resume_from_step = int(resume_audit["step"])
        train_cfg.resume_from_step = resume_from_step
        preflight["train_config"] = asdict(train_cfg)
        preflight["resume_audit"] = resume_audit
        write_json(run_dir / "resume_audit.json", resume_audit)
        write_json(run_dir / "formal_preflight.json", preflight)
        write_json(run_dir / "config.json", {"train": asdict(train_cfg), "data": data_cfg.to_dict(), "model": model_cfg.to_dict()})
    with log_path.open("a") as log:
        log.write(json.dumps({"event": "start", "time": now(), "run_dir": str(run_dir), **preflight}, ensure_ascii=False) + "\n")
        last = time.time()
        global_start = time.time()
        for local_step in range(1, args.steps + 1):
            step = resume_from_step + local_step
            optimizer.zero_grad(set_to_none=True)
            metric_sum: dict[str, float] = {}
            meta_last: dict[str, Any] = {}
            for _ in range(args.grad_accum):
                batch, meta = builder.next_batch(device=device, dtype=torch_dtype(args.dtype))
                out = model.forward_stage2(batch, lambda_pose=args.lambda_pose)
                loss = out["loss"] / args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
                loss.backward()
                for key in ("loss", "loss_visual", "loss_pose", "loss_3d"):
                    value = float(out[key].detach().float().cpu().item())
                    metric_sum[key] = metric_sum.get(key, 0.0) + value / args.grad_accum
                meta_last = meta
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if local_step % args.log_every == 0 or local_step == 1:
                now_time = time.time()
                record = {
                    "event": "train",
                    "time": now(),
                    "step": step,
                    "local_step": local_step,
                    "resume_from_step": resume_from_step,
                    "seconds_per_step_window": (now_time - last) / max(args.log_every, 1),
                    "seconds_total": now_time - global_start,
                    "lr": optimizer.param_groups[0]["lr"],
                    "history_iw": meta_last.get("history_iw"),
                    "history_micro": meta_last.get("history_micro"),
                    "pose_valid_frames": meta_last.get("pose_valid_frames"),
                    "empty_hist_count": meta_last.get("empty_hist_count"),
                    "empty_cur_count": meta_last.get("empty_cur_count"),
                    "datasets": meta_last.get("datasets"),
                    **{f"train/{k}": v for k, v in metric_sum.items()},
                }
                if device.type == "cuda":
                    record["cuda_max_memory_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                last = now_time
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, step)
            if local_step % args.save_every == 0 or local_step == args.steps:
                ckpt = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    run_dir=run_dir,
                    step=step,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    data_cfg=data_cfg,
                    wan_audit=wan_audit,
                )
                log.write(json.dumps({"event": "checkpoint", "time": now(), "step": step, "path": str(ckpt)}, ensure_ascii=False) + "\n")
                log.flush()
        writer.close()
    print(json.dumps({"status": "ok", "run_dir": str(run_dir), "log": str(log_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
