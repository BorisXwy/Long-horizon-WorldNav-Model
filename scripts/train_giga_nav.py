#!/usr/bin/env python3
"""Train the complete GigaNav navigation ablation on R2R."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giga_nav import GigaNavConfig, GigaNavDataConfig, GigaNavModel, GigaNavR2RBatchBuilder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="/sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, default=ROOT / "log/giga_nav_wan21_13b")
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--grad-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_cfg = GigaNavConfig(backbone_checkpoint=args.backbone, action_horizon=args.action_horizon)
    data_cfg = GigaNavDataConfig(action_horizon=args.action_horizon, batch_size=args.batch_size, max_episodes=args.max_episodes)
    data = GigaNavR2RBatchBuilder(data_cfg)
    model = GigaNavModel(model_cfg)
    init_audit = model.load_wan_checkpoint()
    model.to(device=device, dtype=dtype).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    start_step = 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
    report = model.structural_report()
    report["initialization_audit"] = init_audit
    report["data_summary"] = data.summary()
    if args.grad_accumulation_steps < 1:
        raise ValueError("--grad-accumulation-steps must be >= 1")
    report["optimizer"] = {"name": "AdamW", "lr": args.lr, "weight_decay": 1e-2, "physical_batch_size": args.batch_size, "gradient_accumulation_steps": args.grad_accumulation_steps, "effective_batch_size": args.batch_size * args.grad_accumulation_steps}
    (args.output / "config_and_structure.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    with (args.output / "train.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start_step + 1, args.steps + 1):
            tic = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            meta = {}
            loss_value = 0.0
            for micro_step in range(args.grad_accumulation_steps):
                batch, meta = data.next_batch(device=device, dtype=dtype)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                    output = model.forward_policy(obs_latent=batch["obs_latent"], text_embedding=batch["text_embedding"], text_mask=batch["text_mask"], state=batch["state"], action_noise=batch["action_noise"])
                    losses = model.loss(output, batch["action_target"], batch["action_loss_mask"])
                (losses["loss"] / args.grad_accumulation_steps).backward()
                loss_value += float(losses["loss"].detach().cpu()) / args.grad_accumulation_steps
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            optimizer.step()
            record = {"step": step, "loss": loss_value, "grad_norm": float(grad_norm.detach().cpu()), "step_seconds": time.perf_counter() - tic, "micro_steps": args.grad_accumulation_steps, "effective_batch_size": args.batch_size * args.grad_accumulation_steps, "history_micro": meta.get("history_micro"), "sample_ids": meta.get("sample_ids")}
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            if step == 1 or step % 50 == 0:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if step % args.save_interval == 0 or step == args.steps:
                torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "model_config": model_cfg.to_dict(), "data_config": data_cfg.to_dict(), "structure": report}, args.output / f"step_{step:06d}.pt")


if __name__ == "__main__":
    main()
