#!/usr/bin/env python3
"""V1 Stage One training on the real Wan2.1/InfiniteWorld backbone.

This is the first non-scaffold V1 trainer:

  z_obs    -> local memory + Register extractor
  action   -> Wan/InfiniteWorld action encoder condition
  z_future -> RFlow diffusion target

Initialization uses the InfiniteWorld checkpoint, which is the compatible
Wan2.1-1.3B backbone checkpoint with InfiniteWorld's action/HPMC additions.
HPMC is removed at runtime and replaced by the V1 explicit local memory path.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "Infinite-World"))

import infworld.context_parallel.context_parallel_util as cp_util
import infworld.models.dit_model as dit_model_module
from infworld.models.checkpoint import set_grad_checkpoint
from infworld.models.scheduler import RFlowScheduler
from nav.infinite_adapter import InfiniteRegisterAdapter
from nav.v1.data.sharded_hdf5 import V1MultiHdf5Dataset


DEFAULT_DATA = Path("/sharedata/NAV/derived/v1/vae_packs_hdf5_fullgpu14_7x7/workers")
DEFAULT_CKPT = Path("/sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt")
DEFAULT_TEXT = Path("/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt")


def install_non_reentrant_checkpoint() -> None:
    def checkpoint_module(module, *args, **kwargs):
        if getattr(module, "grad_checkpointing", False):
            return torch.utils.checkpoint.checkpoint(
                module, *args, use_reentrant=False, **kwargs
            )
        return module(*args, **kwargs)

    dit_model_module.auto_grad_checkpoint = checkpoint_module


def parse_weights(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, value = part.split(":", 1)
        out[key.strip()] = float(value)
    return out


def worker_roots(root: Path) -> list[Path]:
    roots = sorted(Path(root).glob("worker-*-of-*"))
    if roots:
        return roots
    if (root / "meta" / "index.parquet").exists():
        return [root]
    raise FileNotFoundError(f"找不到 V1 HDF5 worker roots: {root}")


def build_weighted_sampler(dataset: V1MultiHdf5Dataset, weights: dict[str, float], *, seed: int):
    sample_weights: list[float] = []
    dataset_counts: dict[str, int] = {}
    for sub in dataset.datasets:
        counts = sub.index["dataset"].value_counts().to_dict()
        dataset_counts.update({k: dataset_counts.get(k, 0) + int(v) for k, v in counts.items()})
    for sub in dataset.datasets:
        for name in sub.index["dataset"].astype(str):
            n = max(dataset_counts.get(name, 1), 1)
            sample_weights.append(float(weights.get(name, 1.0)) / n)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        ),
        dataset_counts,
    )


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [x["sample_id"] for x in batch],
        "dataset": [x["dataset"] for x in batch],
        "z_obs": torch.stack([x["z_obs"] for x in batch]),
        "z_future": torch.stack([x["z_future"] for x in batch]),
        "action_primitive": torch.stack([x["action_primitive"] for x in batch]),
        "action_valid": torch.stack([x["action_valid"] for x in batch]),
    }


def primitive_to_move_view(action_primitive: torch.Tensor, *, target_len: int = 81) -> tuple[torch.Tensor, torch.Tensor]:
    """Map V1 Primitive ids to InfiniteWorld's old move/view token channels.

    Old labels used 0 as no-op, move forward/back/strafe in move, and turn/look
    in view. We fill the first H V1 actions and pad the rest to no-op.
    """

    b, horizon = action_primitive.shape
    move = torch.zeros((b, target_len), dtype=torch.long, device=action_primitive.device)
    view = torch.zeros((b, target_len), dtype=torch.long, device=action_primitive.device)
    h = min(horizon, target_len)
    ids = action_primitive[:, :h].long()
    # V1 Primitive: STOP=1, MOVE_FORWARD=2, MOVE_BACKWARD=3,
    # STRAFE_LEFT=4, STRAFE_RIGHT=5, TURN_LEFT=6, TURN_RIGHT=7,
    # LOOK_UP=8, LOOK_DOWN=9.
    move[:, :h] = torch.where(ids == 2, torch.ones_like(ids), move[:, :h])
    move[:, :h] = torch.where(ids == 3, torch.full_like(ids, 2), move[:, :h])
    move[:, :h] = torch.where(ids == 4, torch.full_like(ids, 3), move[:, :h])
    move[:, :h] = torch.where(ids == 5, torch.full_like(ids, 4), move[:, :h])
    view[:, :h] = torch.where(ids == 6, torch.full_like(ids, 3), view[:, :h])
    view[:, :h] = torch.where(ids == 7, torch.full_like(ids, 4), view[:, :h])
    view[:, :h] = torch.where(ids == 8, torch.ones_like(ids), view[:, :h])
    view[:, :h] = torch.where(ids == 9, torch.full_like(ids, 2), view[:, :h])
    return move, view


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(*, checkpoint: Path, device: torch.device, train_scope: str) -> InfiniteRegisterAdapter:
    model = InfiniteRegisterAdapter(
        variant="latent_prefix",
        model_cfg={
            "model_type": "t2v",
            "dim": 1536,
            "in_channels": 20,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "num_heads": 12,
            "num_layers": 30,
            "out_channels": 16,
            "caption_channels": 4096,
            "model_max_length": 512,
        },
        register_cfg={
            "channels": 16,
            "register_frames": 4,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_update_layers": 2,
            "chunk_time_tokens": 8,
        },
    )
    audit = model.load_infinite_checkpoint(str(checkpoint))
    model._checkpoint_audit = audit  # lightweight run metadata
    if train_scope == "register":
        model.freeze_backbone()
    elif train_scope == "full":
        model.requires_grad_(True)
    else:
        raise ValueError(train_scope)
    model.to(device=device, dtype=torch.bfloat16).train()
    set_grad_checkpoint(model.backbone)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--text-embedding", type=Path, default=DEFAULT_TEXT)
    ap.add_argument("--run-name", default=f"v1-wan-stage1-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--out-root", type=Path, default=ROOT / "log")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--train-scope", choices=["full", "register"], default="full")
    ap.add_argument(
        "--dataset-weights",
        default="dl3dv:0.35,spatialvid:0.40,re10k:0.15,argoverse2:0.10",
    )
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cp_util.dp_rank = cp_util.cp_rank = 0
    cp_util.dp_size = cp_util.cp_size = 1
    install_non_reentrant_checkpoint()
    seed_everything(args.seed)

    device = torch.device(args.device)
    run_dir = args.out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    writer = SummaryWriter(run_dir / "tensorboard")

    roots = worker_roots(args.data_root)
    dataset = V1MultiHdf5Dataset(roots, load_metadata=False)
    sampler, dataset_counts = build_weighted_sampler(
        dataset, parse_weights(args.dataset_weights), seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )
    iterator = iter(loader)

    model = make_model(checkpoint=args.checkpoint, device=device, train_scope=args.train_scope)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = RFlowScheduler(shift=7.0, use_reversed_velocity=True, use_timestep_transform=True)
    text_condition = torch.load(args.text_embedding, map_location="cpu", weights_only=False)
    y_base = text_condition["y"].to(device=device, dtype=torch.bfloat16)
    y_mask_base = text_condition["y_mask"].to(device=device)

    config = {
        "run_name": args.run_name,
        "mode": "v1_stage1_wan_backbone",
        "data_root": str(args.data_root),
        "dataset_len": len(dataset),
        "dataset_counts": dataset_counts,
        "checkpoint": str(args.checkpoint),
        "checkpoint_audit": getattr(model, "_checkpoint_audit", {}),
        "text_embedding": str(args.text_embedding),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_scope": args.train_scope,
        "trainable_parameters": sum(p.numel() for p in trainable_parameters),
        "dtype": "bf16",
        "objective": "InfiniteWorld RFlow loss on V1 z_future, conditioned by V1 z_obs/local memory and pseudo action",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    with log_path.open("w") as log:
        log.write(json.dumps({"event": "start", **config}, ensure_ascii=False) + "\n")

    start = time.perf_counter()
    last = start
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        dataset_seen: dict[str, int] = {}
        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            for name in batch["dataset"]:
                dataset_seen[name] = dataset_seen.get(name, 0) + 1
            z_obs = batch["z_obs"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
            z_future = batch["z_future"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
            action = batch["action_primitive"].to(device=device, non_blocking=True)
            move, view = primitive_to_move_view(action, target_len=81)
            registers = model.register_memory.extract(z_obs)
            # WanModel.forward builds token_ignore_mask over the concatenated
            # condition+target timeline. For latent_prefix V1:
            #   register T=4 + local_memory T=1 + target T=1 -> T_all=6.
            # The scheduler loss only consumes the final target slice.
            ignore_t = model.register_memory.register_frames + z_obs.shape[2] + z_future.shape[2]
            ignore_mask = torch.zeros(
                z_future.shape[0],
                ignore_t,
                z_future.shape[3],
                z_future.shape[4],
                device=device,
                dtype=torch.bool,
            )
            terms = scheduler.training_losses(
                model,
                z_future,
                model_kwargs={
                    "y": y_base.expand(z_future.shape[0], -1, -1, -1),
                    "y_mask": y_mask_base.expand(z_future.shape[0], -1),
                    "registers": registers,
                    "local_latent": z_obs,
                    "move": move,
                    "view": view,
                },
                x_ignore_mask=ignore_mask,
            )
            loss = terms["loss"].mean()
            loss_sum += float(loss.detach().cpu())
            (loss / args.gradient_accumulation_steps).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            mem_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        else:
            mem_alloc = mem_reserved = 0.0
        now = time.perf_counter()
        if step == 1 or step % args.log_every == 0:
            payload = {
                "event": "step",
                "step": step,
                "loss": loss_sum / args.gradient_accumulation_steps,
                "grad_norm": float(grad_norm.detach().cpu()),
                "sec_per_step_window": (now - last) / max(args.log_every if step > 1 else 1, 1),
                "cuda_max_allocated_gb": round(mem_alloc, 3),
                "cuda_max_reserved_gb": round(mem_reserved, 3),
                "micro_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                "datasets": dict(sorted(dataset_seen.items())),
            }
            last = now
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            with log_path.open("a") as log:
                log.write(json.dumps(payload, ensure_ascii=False) + "\n")
            writer.add_scalar("train/loss", payload["loss"], step)
            writer.add_scalar("train/grad_norm", payload["grad_norm"], step)
            writer.add_scalar("system/sec_per_step", payload["sec_per_step_window"], step)
            writer.add_scalar("system/cuda_max_allocated_gb", payload["cuda_max_allocated_gb"], step)
        if args.save_every > 0 and step % args.save_every == 0:
            torch.save(
                {
                    "backbone": model.backbone.state_dict(),
                    "register_memory": model.register_memory.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                },
                run_dir / f"step-{step:06d}.pt",
            )
        if args.smoke and step >= args.steps:
            break

    total = time.perf_counter() - start
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "register_memory": model.register_memory.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args.steps,
            "config": config,
        },
        run_dir / "final.pt",
    )
    with log_path.open("a") as log:
        log.write(json.dumps({"event": "complete", "steps": args.steps, "total_sec": total}, ensure_ascii=False) + "\n")
    writer.close()


if __name__ == "__main__":
    main()
