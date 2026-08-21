#!/usr/bin/env python3
"""Final Stage2 cotrain with layer-16 dense camera-query 3D head.

This continues from an existing final Stage2 checkpoint, keeps the same video
generation/register/action data path, replaces the attached mean-pool pose head
with a spatial camera-query readout, and reads current-observation hidden states
from a selected Wan block layer.
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
sys.path.insert(0, str(ROOT / "scripts"))

from nav.v1.stage2_final import FinalStage2BatchBuilder, FinalStage2DataConfig, FinalStage2WanConfig, FinalStage2WanModel  # noqa: E402
from train_v1_stage2_final_cotrain import install_non_reentrant_wan_checkpoint, now, torch_dtype, write_json  # noqa: E402


DEFAULT_CHECKPOINT = Path(
    "/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/log/v1_stage2_final_cotrain/"
    "stage2_final_branchmask_policyreg_venv_iw14816_20260819_015948/checkpoints/step_002000.pt"
)


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
    save_every: int
    log_every: int
    seed: int
    checkpoint: str
    pose_head_type: str
    pose_readout_layer: int
    output_root: str


def make_dataclass(cls, payload: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    clean = {k: v for k, v in payload.items() if k in allowed}
    for key in ("manifest", "latent_root", "text_empty", "text_cache_root", "re10k_camera_root"):
        if key in clean:
            clean[key] = Path(clean[key])
    if cls is FinalStage2WanConfig and "register_condition_grid" in clean:
        clean["register_condition_grid"] = tuple(clean["register_condition_grid"])
    return cls(**clean)


def load_model_from_stage2_checkpoint(
    checkpoint: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    pose_head_type: str,
    pose_readout_layer: int,
) -> tuple[FinalStage2WanModel, dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = make_dataclass(FinalStage2WanConfig, ckpt["model_config"])
    cfg.pose_head_type = pose_head_type
    cfg.pose_readout_layer = int(pose_readout_layer)
    model = FinalStage2WanModel(cfg)
    model.remove_hmpc()
    target = model.state_dict()
    source = ckpt["model"]
    compatible = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    skipped = [
        {"key": k, "source": list(v.shape), "target": list(target[k].shape) if k in target else None}
        for k, v in source.items()
        if k not in compatible
    ][:128]
    result = model.load_state_dict(compatible, strict=False)
    audit = {
        "checkpoint": str(checkpoint),
        "loaded_compatible_keys": len(compatible),
        "missing_keys_count": len(result.missing_keys),
        "unexpected_keys_count": len(result.unexpected_keys),
        "missing_keys_head": result.missing_keys[:64],
        "unexpected_keys_head": result.unexpected_keys[:64],
        "skipped_head": skipped,
        "pose_head_reinitialized": True,
        "pose_head_type": pose_head_type,
        "pose_readout_layer": pose_readout_layer,
    }
    return model.to(device=device, dtype=dtype).train(), ckpt, audit


def save_checkpoint(
    *,
    model: FinalStage2WanModel,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    step: int,
    train_cfg: TrainConfig,
    model_cfg: FinalStage2WanConfig,
    data_cfg: FinalStage2DataConfig,
    source_audit: dict[str, Any],
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
            "data_config": data_cfg.to_dict(),
            "source_checkpoint_audit": source_audit,
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"stage2_probe_layer16_cotrain_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-pose", type=float, default=0.1)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--pose-readout-layer", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=ROOT / "log" / "v1_stage2_final_probe_layer16_cotrain")
    args = parser.parse_args()

    if args.batch_size * args.grad_accum != 16:
        raise SystemExit("final Stage2 probe cotrain requires effective batch size = 16")
    os.environ.setdefault("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_non_reentrant_wan_checkpoint()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    model, source_ckpt, source_audit = load_model_from_stage2_checkpoint(
        args.checkpoint,
        device=device,
        dtype=dtype,
        pose_head_type="dense_camera_query",
        pose_readout_layer=args.pose_readout_layer,
    )
    data_cfg = make_dataclass(FinalStage2DataConfig, source_ckpt["data_config"])
    data_cfg.batch_size = args.batch_size
    data_cfg.seed = args.seed
    builder = FinalStage2BatchBuilder(data_cfg)
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
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        checkpoint=str(args.checkpoint),
        pose_head_type=model.cfg.pose_head_type,
        pose_readout_layer=model.cfg.pose_readout_layer,
        output_root=str(args.output_root),
    )
    preflight = {
        "event": "stage2_probe_layer16_cotrain_preflight",
        "time": now(),
        "train": asdict(train_cfg),
        "data": builder.summary(),
        "model": model.structural_report(),
        "source_checkpoint": source_audit,
        "loss": "L_stage2 = L_visual + lambda_pose * L_pose; pose reads selected Wan layer hidden with DenseCameraQueryPoseHead",
    }
    write_json(run_dir / "formal_preflight.json", preflight)
    write_json(run_dir / "config.json", {"train": asdict(train_cfg), "data": data_cfg.to_dict(), "model": model.cfg.to_dict()})
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
            metric_sum: dict[str, float] = {}
            meta_last: dict[str, Any] = {}
            for _ in range(args.grad_accum):
                batch, meta = builder.next_batch(device=device, dtype=dtype)
                out = model.forward_stage2(batch, lambda_pose=args.lambda_pose)
                loss = out["loss"] / args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
                loss.backward()
                for key in ("loss", "loss_visual", "loss_pose", "loss_3d"):
                    metric_sum[key] = metric_sum.get(key, 0.0) + float(out[key].detach().float().cpu().item()) / args.grad_accum
                meta_last = meta
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
                    "history_iw": meta_last.get("history_iw"),
                    "history_micro": meta_last.get("history_micro"),
                    "pose_valid_frames": meta_last.get("pose_valid_frames"),
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
            if step % args.save_every == 0 or step == args.steps:
                ckpt = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    run_dir=run_dir,
                    step=step,
                    train_cfg=train_cfg,
                    model_cfg=model.cfg,
                    data_cfg=data_cfg,
                    source_audit=source_audit,
                )
                log.write(json.dumps({"event": "checkpoint", "time": now(), "step": step, "path": str(ckpt)}, ensure_ascii=False) + "\n")
                log.flush()
    writer.close()
    print(json.dumps({"status": "ok", "run_dir": str(run_dir), "log": str(log_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
