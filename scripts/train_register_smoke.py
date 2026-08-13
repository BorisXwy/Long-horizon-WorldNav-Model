#!/usr/bin/env python3
"""Register 模块最小训练；用于在全量 DiT 训练前验证状态递归和日志链路。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nav.register_memory import RegisterMemory
from nav.spatial_register_memory import SpatialRegisterMemory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["latent_prefix", "dit_condition"], required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name")
    parser.add_argument("--latent-cache", type=Path)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(args.device)
    run_name = args.run_name or f"{args.variant}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = ROOT / "log" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(run_dir / "tensorboard")

    model = (
        SpatialRegisterMemory()
        if args.variant == "latent_prefix"
        else RegisterMemory(caption_channels=4096)
    ).to(device)
    # 只用于验证 Register 能学习上一 chunk -> 下一 chunk 的状态摘要。
    predictor = nn.Identity() if args.variant == "latent_prefix" else nn.Linear(4096, 16).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(predictor.parameters()),
        lr=1e-5,
        weight_decay=0.01,
    )
    log_path = run_dir / "train.log"
    with log_path.open("w") as log:
        for step in range(1, args.steps + 1):
            if args.latent_cache:
                cached = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
                chunks = cached["chunks"]
                # cache: [1,2,C,T,H,W]
                previous = chunks[:, 0].to(device)
                following = chunks[:, 1].to(device)
            else:
                previous = torch.randn(1, 16, 21, 28, 56, device=device)
                following = 0.7 * previous + 0.3 * torch.randn_like(previous)
            registers = model.extract(previous)
            if args.variant == "latent_prefix":
                prediction = registers.mean(dim=(2, 3, 4))
            else:
                injected = model.as_dit_condition(registers)
                prediction = predictor(injected.mean(dim=1))
            target = following.mean(dim=(2, 3, 4))
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            payload = {
                "step": step,
                "loss": float(loss.detach()),
                "register_grad_norm": float(grad_norm),
                "variant": args.variant,
            }
            line = json.dumps(payload)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            writer.add_scalar("train/loss", payload["loss"], step)
            writer.add_scalar("train/register_grad_norm", payload["register_grad_norm"], step)
    torch.save(
        {"register_memory": model.state_dict(), "predictor": predictor.state_dict()},
        run_dir / "checkpoint-final.pt",
    )
    writer.close()
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
