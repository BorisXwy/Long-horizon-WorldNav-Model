#!/usr/bin/env python3
"""YAML-driven full Wan2.1 GigaNav policy/video/cotrain entrypoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giga_nav import (  # noqa: E402
    GigaNavAlignedWorldActionBatchBuilder,
    GigaNavConfig,
    GigaNavDataConfig,
    GigaNavModel,
    GigaNavR2RBatchBuilder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("policy_only", "video_only", "cotrain"))
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return payload


def _paths(data: dict) -> dict:
    result = dict(data)
    for key in ("latent_manifest_dir", "rendered_manifest", "text_empty", "text_cache_root"):
        if key in result:
            result[key] = Path(result[key])
    return result


def _build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = str(cfg.get("name", "adamw")).lower()
    kwargs = {
        "lr": float(cfg.get("lr", 6e-5)),
        "weight_decay": float(cfg.get("weight_decay", 1e-2)),
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adafactor":
        # PyTorch Adafactor keeps factored second moments and is used here as
        # the stable low-memory counterpart of GWP-0.5's CAME8Bit.  The local
        # public CAME8Bit implementation materializes hundreds of thousands of
        # Python quantization blocks for a 1.3B model and is not operationally
        # suitable for this single-GPU run.
        return torch.optim.Adafactor(model.parameters(), foreach=False, **kwargs)
    raise ValueError(f"unsupported optimizer: {name}")


def main() -> None:
    args = parse_args()
    raw = _load_yaml(args.config)
    mode = args.mode or str(raw["experiment"]["mode"])
    if mode not in {"policy_only", "video_only", "cotrain"}:
        raise ValueError(f"unknown mode={mode!r}")
    train_cfg = dict(raw["train"])
    objective_cfg = dict(raw["objective"])
    device = torch.device(args.device or train_cfg.get("device", "cuda:0"))
    steps = int(args.steps or train_cfg.get("steps", 5000))
    output = args.output or ROOT / str(train_cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    seed = int(train_cfg.get("seed", 20260907))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_values = dict(raw.get("model", {}))
    if "backbone_checkpoint" in model_values:
        model_values["backbone_checkpoint"] = Path(model_values["backbone_checkpoint"])
    model_cfg = GigaNavConfig(**model_values)
    data_cfg = GigaNavDataConfig(**_paths(raw.get("data", {})))
    alignment = str(raw["experiment"].get("alignment", "causal_h8_video"))
    if alignment == "legacy_policy_h8":
        if mode != "policy_only":
            raise ValueError("legacy_policy_h8 is valid only for policy_only")
        data = GigaNavR2RBatchBuilder(data_cfg)
    elif alignment == "causal_h8_video":
        data = GigaNavAlignedWorldActionBatchBuilder(data_cfg)
    else:
        raise ValueError(f"unknown alignment={alignment!r}")

    model = GigaNavModel(model_cfg)
    init_audit = model.load_wan_checkpoint()
    model.to(device=device, dtype=dtype).train()
    model.policy_head.float()
    optimizer = _build_optimizer(model, raw.get("optimizer", {}))
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 32))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    physical_bs = int(data_cfg.batch_size)
    if physical_bs * accumulation != int(train_cfg.get("effective_batch_size", 32)):
        raise ValueError("physical batch * gradient accumulation must equal configured effective_batch_size")

    run_report = {
        "schema": "giga_nav_multitask_v1",
        "config_path": str(args.config.resolve()),
        "mode": mode,
        "alignment": alignment,
        "model": model.structural_report(),
        "initialization_audit": init_audit,
        "data": data.summary(),
        "yaml": raw,
    }
    (output / "config_and_structure.json").write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    writer = SummaryWriter(str(output / "tensorboard"))
    rng = random.Random(seed + 1)
    acwm_probability = float(objective_cfg.get("acwm_probability", 0.5))
    video_weight = float(objective_cfg.get("video_weight", 1.0))
    policy_weight = float(objective_cfg.get("policy_weight", 5.0))
    save_interval = int(train_cfg.get("save_interval", 1000))
    log_interval = int(train_cfg.get("log_interval", 10))

    with (output / "train.jsonl").open("a", encoding="utf-8") as log:
        for step in range(1, steps + 1):
            tic = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            totals = {"loss": 0.0, "loss_video_flow": 0.0, "loss_policy_ce": 0.0}
            branch_counts = {"acwm": 0, "wam": 0, "policy": 0}
            meta: dict = {}
            for _ in range(accumulation):
                batch, meta = data.next_batch(device=device, dtype=dtype)
                if mode == "policy_only" and alignment == "legacy_policy_h8":
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                        output_dict = model.forward_policy(
                            obs_latent=batch["obs_latent"],
                            text_embedding=batch["text_embedding"],
                            text_mask=batch["text_mask"],
                            state=batch["state"],
                            action_noise=batch["action_noise"],
                        )
                        losses = model.loss(output_dict, batch["action_target"], batch["action_loss_mask"])
                    losses = {
                        "loss": losses["loss"],
                        "loss_video_flow": losses["loss"] * 0.0,
                        "loss_policy_ce": losses["loss_policy_ce"],
                    }
                    branch = "policy"
                else:
                    flow = model.flow_sample(batch["future_latent"])
                    if mode == "video_only":
                        branch = "acwm"
                    elif mode == "policy_only":
                        branch = "policy"
                    else:
                        branch = "acwm" if rng.random() < acwm_probability else "wam"
                    include_video = branch in {"acwm", "wam"}
                    include_policy = branch in {"policy", "wam"}
                    if branch == "acwm":
                        action_input = model.action_classes_to_input(batch["action_target"])
                    else:
                        action_input = torch.zeros(
                            batch["action_target"].shape[0],
                            model_cfg.action_horizon,
                            model_cfg.action_input_dim,
                            device=device,
                            dtype=dtype,
                        )
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                        output_dict = model.forward_world_action(
                            reference_latent=batch["reference_latent"],
                            future_noisy=flow["noisy"],
                            visual_timestep=flow["timestep"],
                            text_embedding=batch["text_embedding"],
                            text_mask=batch["text_mask"],
                            action_input=action_input,
                            state=batch["state"],
                            return_video=include_video,
                        )
                        losses = model.multitask_loss(
                            output_dict,
                            video_target=flow["target_velocity"] if include_video else None,
                            action_target=batch["action_target"],
                            action_mask=batch["action_loss_mask"],
                            include_video=include_video,
                            include_policy=include_policy,
                            video_weight=video_weight,
                            policy_weight=policy_weight,
                        )
                (losses["loss"] / accumulation).backward()
                branch_counts[branch] += 1
                for key in totals:
                    totals[key] += float(losses[key].detach().cpu()) / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            elapsed = time.perf_counter() - tic
            record = {
                "step": step,
                **totals,
                "grad_norm": float(grad_norm.detach().cpu()),
                "step_seconds": elapsed,
                "effective_batch_size": physical_bs * accumulation,
                "branch_counts": branch_counts,
                "history_micro": meta.get("history_micro"),
                "sample_ids": meta.get("sample_ids"),
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            for key, value in totals.items():
                writer.add_scalar(f"train/{key}", value, step)
            writer.add_scalar("train/grad_norm", record["grad_norm"], step)
            writer.add_scalar("train/step_seconds", elapsed, step)
            for key, value in branch_counts.items():
                writer.add_scalar(f"train/branch_count_{key}", value, step)
            writer.flush()
            if step == 1 or step % log_interval == 0:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if step % save_interval == 0 or step == steps:
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": model_cfg.to_dict(),
                        "data_config": data_cfg.to_dict(),
                        "run": run_report,
                    },
                    output / f"step_{step:06d}.pt",
                )
    writer.close()


if __name__ == "__main__":
    main()
