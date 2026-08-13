#!/usr/bin/env python3
"""Train/smoke the V1 Stage One scaffold.

This trainer validates the V1 sparse pack contract:
  z_obs [B,16,1,56,112]
  z_future [B,16,1,56,112]
  action_primitive [B,4]

It intentionally uses a small Transformer scaffold, not the full Wan DiT.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nav.v1.data.sharded_hdf5 import V1MultiHdf5Dataset
from nav.v1.models.stage1 import V1StageOneScaffold


DEFAULT_DATA = Path("/sharedata/NAV/derived/v1/vae_packs_hdf5_fullgpu14_7x7/workers")


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--run-name", default=f"v1-stage1-scaffold-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--out-root", type=Path, default=ROOT / "log")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lambda-visual", type=float, default=1.0)
    ap.add_argument("--lambda-action", type=float, default=0.2)
    ap.add_argument("--noise-sigma", type=float, default=1.0)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--register-tokens", type=int, default=64)
    ap.add_argument("--action-horizon", type=int, default=4)
    ap.add_argument(
        "--dataset-weights",
        default="dl3dv:0.35,spatialvid:0.40,re10k:0.15,argoverse2:0.10",
    )
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    run_dir = args.out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    roots = worker_roots(args.data_root)
    dataset = V1MultiHdf5Dataset(roots, load_metadata=False)
    sampler, dataset_counts = build_weighted_sampler(
        dataset,
        parse_weights(args.dataset_weights),
        seed=args.seed,
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

    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_available() and device.type == "cuda" else torch.float32
    model = V1StageOneScaffold(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_register_tokens=args.register_tokens,
        action_horizon=args.action_horizon,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tensorboard"))
    except Exception:
        writer = None

    param_count = sum(p.numel() for p in model.parameters())
    config = {
        "run_name": args.run_name,
        "data_root": str(args.data_root),
        "worker_roots": [str(x) for x in roots],
        "dataset_len": len(dataset),
        "dataset_counts": dataset_counts,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_visual": args.lambda_visual,
        "lambda_action": args.lambda_action,
        "noise_sigma": args.noise_sigma,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "register_tokens": args.register_tokens,
        "action_horizon": args.action_horizon,
        "trainable_parameters": param_count,
        "dtype": str(dtype),
        "note": "V1 Stage One scaffold; not full Wan/InfiniteWorld DiT.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    with log_path.open("a") as log:
        log.write(json.dumps({"event": "start", **config}, ensure_ascii=False) + "\n")

    model.train()
    start = time.perf_counter()
    last = start
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        z_obs = batch["z_obs"].to(device=device, dtype=dtype, non_blocking=True)
        z_future = batch["z_future"].to(device=device, dtype=dtype, non_blocking=True)
        action = batch["action_primitive"][:, : args.action_horizon].to(
            device=device,
            non_blocking=True,
        )
        valid = batch["action_valid"][:, : args.action_horizon].to(
            device=device,
            non_blocking=True,
        )

        noise = torch.randn_like(z_future) * args.noise_sigma
        visual_t = torch.rand((z_future.shape[0],), device=device)
        z_noisy = z_future + visual_t[:, None, None, None, None].to(dtype) * noise

        optimizer.zero_grad(set_to_none=True)
        out = model(
            z_obs=z_obs,
            z_future_noisy=z_noisy,
            visual_timestep=visual_t,
        )
        pred_noise = out["pred_noise"]
        visual_loss = F.mse_loss(pred_noise.float(), noise.float())
        logits = out["primitive_logits"].float()
        action_loss_all = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            action.reshape(-1),
            reduction="none",
        ).view_as(action)
        action_loss = (action_loss_all * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        loss = args.lambda_visual * visual_loss + args.lambda_action * action_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            mem_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        else:
            mem_alloc = 0.0
            mem_reserved = 0.0

        now = time.perf_counter()
        if step == 1 or step % args.log_every == 0:
            dt = now - last
            last = now
            payload = {
                "event": "step",
                "step": step,
                "loss": float(loss.detach().cpu()),
                "visual_loss": float(visual_loss.detach().cpu()),
                "action_loss": float(action_loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "sec_per_step_window": dt / max(args.log_every if step > 1 else 1, 1),
                "cuda_max_allocated_gb": round(mem_alloc, 3),
                "cuda_max_reserved_gb": round(mem_reserved, 3),
                "datasets": dict(sorted({x: batch["dataset"].count(x) for x in set(batch["dataset"])}.items())),
            }
            with log_path.open("a") as log:
                log.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            if writer is not None:
                writer.add_scalar("train/loss", payload["loss"], step)
                writer.add_scalar("train/visual_loss", payload["visual_loss"], step)
                writer.add_scalar("train/action_loss", payload["action_loss"], step)
                writer.add_scalar("train/grad_norm", payload["grad_norm"], step)
                writer.add_scalar("system/cuda_max_allocated_gb", payload["cuda_max_allocated_gb"], step)
                writer.add_scalar("system/sec_per_step", payload["sec_per_step_window"], step)

        if args.save_every > 0 and step % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": config},
                run_dir / f"step-{step:06d}.pt",
            )

        if args.smoke and step >= args.steps:
            break

    total_time = time.perf_counter() - start
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": args.steps, "config": config},
        run_dir / "final.pt",
    )
    with log_path.open("a") as log:
        log.write(json.dumps({"event": "complete", "steps": args.steps, "total_sec": total_time}, ensure_ascii=False) + "\n")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
