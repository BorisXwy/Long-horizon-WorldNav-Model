#!/usr/bin/env python3
"""Full-pipeline smoke test for the formal V1 WorldNav model.

This is not a toy-path test.  It uses ``V1FullWorldNavModel`` and runs
the same formal data-flow surfaces used by Stage One/Two/Three:

- Stage One: video generation memory loss;
- Stage Two: Stage One loss + 3D probe loss;
- Stage Three: policy/action flow loss;
- inference: videogen latent denoise path and policy action path.

The tensors are synthetic by default so the test is cheap and independent from
large datasets/weights.  Model parameters are randomly initialized on purpose;
the objective is chain correctness, gradient/update integrity and structural
audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nav.v1.models.full_model import V1FullModelConfig, V1FullWorldNavModel


GROUP_PREFIXES = (
    "visual_stem",
    "action_encoder",
    "register_cell",
    "backbone",
    "future_head",
    "action_decoder",
    "geometry_probe",
)


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def make_batch(
    cfg: V1FullModelConfig,
    *,
    batch_size: int,
    history_steps: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    text_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(20260814)
    shape = (batch_size, cfg.latent_channels, latent_t, latent_h, latent_w)
    return {
        "history_latents": torch.randn(
            batch_size,
            history_steps,
            cfg.latent_channels,
            latent_t,
            latent_h,
            latent_w,
            generator=g,
            device=device,
        ),
        "a_hist_primitives": torch.randint(
            0,
            cfg.num_primitives,
            (batch_size, history_steps, cfg.action_horizon),
            generator=g,
            device=device,
        ),
        "z_obs": torch.randn(*shape, generator=g, device=device),
        "z_future_noisy": torch.randn(*shape, generator=g, device=device),
        "z_future_target": torch.randn(*shape, generator=g, device=device),
        "visual_timestep": torch.rand(batch_size, generator=g, device=device),
        "a_cur_primitives": torch.randint(
            0,
            cfg.num_primitives,
            (batch_size, cfg.action_horizon),
            generator=g,
            device=device,
        ),
        "a_noise": torch.randn(
            batch_size,
            cfg.action_horizon,
            cfg.action_dim,
            generator=g,
            device=device,
        ),
        "action_target": torch.randn(
            batch_size,
            cfg.action_horizon,
            cfg.action_dim,
            generator=g,
            device=device,
        ),
        "action_primitives": torch.randint(
            0,
            cfg.num_primitives,
            (batch_size, cfg.action_horizon),
            generator=g,
            device=device,
        ),
        "action_timestep": torch.rand(batch_size, generator=g, device=device),
        "geometry_target": torch.randn(batch_size, cfg.geometry_dim, generator=g, device=device),
        "text_tokens": torch.randn(batch_size, text_tokens, cfg.hidden_dim, generator=g, device=device),
    }


def group_name(param_name: str) -> str:
    for prefix in GROUP_PREFIXES:
        if param_name.startswith(prefix + ".") or param_name == prefix:
            return prefix
    return "other"


def snapshot_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}


def summarize_grads(model: torch.nn.Module) -> dict[str, float]:
    sums: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        group = group_name(name)
        value = float(param.grad.detach().float().norm().item())
        sums[group] = sums.get(group, 0.0) + value
    return {k: v for k, v in sorted(sums.items())}


def summarize_changes(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
    changes: dict[str, float] = {}
    for name, param in model.named_parameters():
        if name not in before:
            continue
        group = group_name(name)
        delta = (param.detach() - before[name].to(param.device)).float().abs().max().item()
        changes[group] = max(changes.get(group, 0.0), float(delta))
    return {k: v for k, v in sorted(changes.items())}


def assert_finite_loss(loss: torch.Tensor, stage: str) -> None:
    if not torch.isfinite(loss):
        raise AssertionError(f"{stage} loss is not finite: {loss}")


def train_one_stage(
    *,
    model: V1FullWorldNavModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    name: str,
    fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    expected_changed: tuple[str, ...],
) -> dict[str, object]:
    model.train()
    before = snapshot_params(model)
    optimizer.zero_grad(set_to_none=True)
    out = fn(batch)
    loss = out["loss"]
    assert_finite_loss(loss, name)
    loss.backward()
    grad_norms = summarize_grads(model)
    optimizer.step()
    changes = summarize_changes(model, before)
    changed_groups = sorted(group for group, value in changes.items() if value > 0.0)
    missing = [group for group in expected_changed if changes.get(group, 0.0) <= 0.0]
    if missing:
        raise AssertionError(f"{name} expected parameter groups did not update: {missing}; changes={changes}")
    return {
        "loss": float(loss.detach().cpu().item()),
        "grad_norms": grad_norms,
        "max_abs_param_change": changes,
        "changed_groups": changed_groups,
        "output_shapes": {
            key: list(value.shape)
            for key, value in out.items()
            if isinstance(value, torch.Tensor) and key != "loss"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--register-tokens", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent-t", type=int, default=2)
    parser.add_argument("--latent-h", type=int, default=8)
    parser.add_argument("--latent-w", type=int, default=8)
    parser.add_argument("--text-tokens", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default=str(ROOT / "log" / "full_pipeline_smoke"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(20260814)
    cfg = V1FullModelConfig(
        hidden_dim=args.hidden_dim,
        num_register_tokens=args.register_tokens,
        num_register_layers=args.layers,
        num_backbone_layers=args.layers,
        num_heads=args.heads,
        patch_size=(1, 2, 2),
    )
    model = V1FullWorldNavModel(cfg).to(device)
    batch = make_batch(
        cfg,
        batch_size=args.batch_size,
        history_steps=args.history_steps,
        latent_t=args.latent_t,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        text_tokens=args.text_tokens,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    audit = model.structural_audit()
    if not audit["uses_register_cell"]:
        raise AssertionError("V1FullWorldNavModel must use RegisterCell")
    if audit["has_register_extractor_module"] or audit["has_register_updater_module"]:
        raise AssertionError(f"Extractor/Updater modules are not allowed in formal smoke: {audit}")
    if audit["has_additive_action_bias_path"]:
        raise AssertionError(f"Additive action bias path detected: {audit}")

    stage1 = train_one_stage(
        model=model,
        optimizer=optimizer,
        batch=batch,
        name="stage1",
        fn=model.forward_stage1,
        expected_changed=("visual_stem", "action_encoder", "register_cell", "backbone", "future_head"),
    )
    stage2 = train_one_stage(
        model=model,
        optimizer=optimizer,
        batch=batch,
        name="stage2",
        fn=lambda b: model.forward_stage2(b, lambda_3d=0.1),
        expected_changed=("visual_stem", "action_encoder", "register_cell", "backbone", "future_head", "geometry_probe"),
    )
    stage3 = train_one_stage(
        model=model,
        optimizer=optimizer,
        batch=batch,
        name="stage3",
        fn=lambda b: model.forward_stage3(b, lambda_ce=0.1, lambda_video=0.25, lambda_3d=0.05),
        expected_changed=(
            "visual_stem",
            "action_encoder",
            "register_cell",
            "backbone",
            "future_head",
            "action_decoder",
            "geometry_probe",
        ),
    )

    model.eval()
    videogen = model.infer_videogen(batch)
    policy = model.infer_policy(batch)
    if videogen["z_future"].shape != batch["z_future_noisy"].shape:
        raise AssertionError("videogen output shape mismatch")
    if policy["action_chunk"].shape != batch["a_noise"].shape:
        raise AssertionError("policy action output shape mismatch")
    if policy["primitive_ids"].shape != batch["action_primitives"].shape:
        raise AssertionError("policy primitive shape mismatch")
    if not all(math.isfinite(x["loss"]) for x in (stage1, stage2, stage3)):
        raise AssertionError("non-finite stage loss")

    report = {
        "event": "v1_full_pipeline_smoke_complete",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "config": cfg.to_dict(),
        "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "structural_audit": audit,
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "inference": {
            "videogen_z_future": list(videogen["z_future"].shape),
            "videogen_future_velocity": list(videogen["future_velocity"].shape),
            "policy_action_chunk": list(policy["action_chunk"].shape),
            "policy_primitive_logits": list(policy["primitive_logits"].shape),
            "policy_primitive_ids": list(policy["primitive_ids"].shape),
        },
    }

    run_dir = Path(args.output_dir) / f"v1_full_pipeline_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "report": str(report_path), **report["inference"]}, indent=2))


if __name__ == "__main__":
    main()
