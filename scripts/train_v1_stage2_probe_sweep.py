#!/usr/bin/env python3
"""Frozen-backbone Stage2 3D probe sweep over Wan DiT layers.

This experiment answers a narrow question:

  Given a trained final-standard Stage2 checkpoint, which Wan block hidden
  layer contains the most linearly/readout-accessible pose information?

The script keeps the complete Stage2 model path intact, but freezes it.  It
installs forward hooks on selected Wan blocks, extracts the current ``Z_obs``
visual prefix tokens from each hooked layer, and trains one dense
camera-query pose probe per layer.

This is not a scaffold path: data, checkpoint loading, Register rollout,
condition/action tokens, and Wan forward are the same final Stage2 code used by
``train_v1_stage2_final_cotrain.py`` / ``eval_v1_stage2_final_cotrain.py``.
Only the optimization target is restricted to the newly-created probe heads.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
INF_WORLD_ROOT = Path(os.environ.get("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World")))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(INF_WORLD_ROOT))

from nav.v1.stage2_final import FinalStage2BatchBuilder, FinalStage2DataConfig, FinalStage2WanConfig, FinalStage2WanModel  # noqa: E402


DEFAULT_CHECKPOINT = Path(
    "/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/log/v1_stage2_final_cotrain/"
    "stage2_final_branchmask_policyreg_venv_iw14816_20260819_015948/checkpoints/step_002000.pt"
)


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


def make_dataclass(cls, payload: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    clean = {k: v for k, v in payload.items() if k in allowed}
    for key in ("manifest", "latent_root", "text_empty", "text_cache_root", "re10k_camera_root"):
        if key in clean:
            clean[key] = Path(clean[key])
    if cls is FinalStage2WanConfig and "register_condition_grid" in clean:
        clean["register_condition_grid"] = tuple(clean["register_condition_grid"])
    return cls(**clean)


def load_frozen_model(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> tuple[FinalStage2WanModel, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = make_dataclass(FinalStage2WanConfig, ckpt["model_config"])
    model = FinalStage2WanModel(cfg)
    model.remove_hmpc()
    result = model.load_state_dict(ckpt["model"], strict=False)
    bad_missing = [key for key in result.missing_keys if not key.startswith("backbone.latent_encoder.")]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing[:20]} unexpected={result.unexpected_keys[:20]}")
    model.to(device=device, dtype=dtype).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


class DenseCameraQueryPoseProbe(nn.Module):
    """VGGT-inspired camera query probe that preserves spatial tokens.

    Per latent frame, a learned camera query cross-attends to all spatial tokens
    before an MLP predicts ``[tx,ty,tz,qw,qx,qy,qz,fov_x,fov_y]``.
    """

    def __init__(self, *, hidden_dim: int, num_heads: int = 8, mlp_ratio: float = 1.0, out_dim: int = 9) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        inner = max(hidden_dim, int(hidden_dim * mlp_ratio))
        self.pose = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, out_dim),
        )

    def forward(self, obs_tokens: torch.Tensor, *, latent_shape: tuple[int, int, int], patch_size: tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
        t, h, w = latent_shape
        pt, ph, pw = patch_size
        grid_t, grid_h, grid_w = t // pt, h // ph, w // pw
        b, n, d = obs_tokens.shape
        expected = grid_t * grid_h * grid_w
        if n != expected:
            raise ValueError(f"obs token count {n} != expected {expected}")
        x = obs_tokens.view(b, grid_t, grid_h * grid_w, d)
        x = self.token_norm(x)
        x = x.reshape(b * grid_t, grid_h * grid_w, d)
        q = self.query_norm(self.query).expand(b * grid_t, -1, -1)
        cam, _ = self.cross_attn(q, x, x, need_weights=False)
        pose = self.pose(cam[:, 0]).view(b, grid_t, -1)
        return pose


def masked_pose_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    return (((pred - target) ** 2) * mask).sum() / mask.expand_as(pred).sum().clamp_min(1.0)


@torch.no_grad()
def pose_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask.bool()
    if valid.sum().item() <= 0:
        return {
            "pose_mse": 0.0,
            "translation_mae": 0.0,
            "rotation_angle_deg": 0.0,
            "fov_mae": 0.0,
            "valid_frames": 0.0,
        }
    pred = pred.float()
    target = target.float()
    pose_mse = masked_pose_mse(pred, target, mask).float()
    pv = pred[valid]
    tv = target[valid]
    trans_mae = (pv[:, :3] - tv[:, :3]).abs().mean()
    pq = pv[:, 3:7] / pv[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tq = tv[:, 3:7] / tv[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rot = 2.0 * torch.acos((pq * tq).sum(dim=-1).abs().clamp(max=1.0)) * 180.0 / torch.pi
    fov = (pv[:, 7:9] - tv[:, 7:9]).abs().mean()
    return {
        "pose_mse": float(pose_mse.detach().cpu().item()),
        "translation_mae": float(trans_mae.detach().cpu().item()),
        "rotation_angle_deg": float(rot.mean().detach().cpu().item()),
        "fov_mae": float(fov.detach().cpu().item()),
        "valid_frames": float(valid.sum().detach().cpu().item()),
    }


class LayerCapture:
    def __init__(self, model: FinalStage2WanModel, layers: list[int]) -> None:
        self.model = model
        self.layers = layers
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles = []
        max_layer = len(model.backbone.blocks) - 1
        for layer in layers:
            if layer < 0 or layer > max_layer:
                raise ValueError(f"layer {layer} out of range 0..{max_layer}")
            self.handles.append(model.backbone.blocks[layer].register_forward_hook(self._make_hook(layer)))

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.outputs[layer] = output.detach().float()
        return hook

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def evaluate_probes(
    *,
    model: FinalStage2WanModel,
    capture: LayerCapture,
    probes: nn.ModuleDict,
    builder: FinalStage2BatchBuilder,
    device: torch.device,
    dtype: torch.dtype,
    layers: list[int],
    batches: int,
) -> dict[int, dict[str, float]]:
    sums = {layer: {"pose_mse": 0.0, "translation_mae": 0.0, "rotation_angle_deg": 0.0, "fov_mae": 0.0, "valid_frames": 0.0, "n": 0.0} for layer in layers}
    for probe in probes.values():
        probe.eval()
    for _ in range(batches):
        batch, _meta = builder.next_batch(device=device, dtype=dtype)
        capture.clear()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            _ = model.forward_core(batch, return_prefix_hidden=False)
        target = batch["pose_target"].to(device=device, dtype=torch.float32)
        mask = batch["pose_mask"].to(device=device)
        latent_shape = (batch["z_obs"].shape[2], batch["z_obs"].shape[3], batch["z_obs"].shape[4])
        with torch.no_grad():
            for layer in layers:
                obs_hidden = model.current_obs_hidden(capture.outputs[layer], batch["z_obs"])
                pred = probes[str(layer)](obs_hidden, latent_shape=latent_shape)
                row = pose_metrics(pred, target, mask)
                for key, value in row.items():
                    sums[layer][key] += value
                sums[layer]["n"] += 1.0
    out: dict[int, dict[str, float]] = {}
    for layer, row in sums.items():
        n = max(row.pop("n"), 1.0)
        out[layer] = {key: value / n for key, value in row.items()}
    for probe in probes.values():
        probe.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-name", default=f"stage2_probe_sweep_step2000_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--layers", default="4,8,12,16,20,24,29")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--dataset-weights", default="re10k=1.0")
    parser.add_argument("--history-iw-chunks", default="1")
    parser.add_argument("--seed", type=int, default=2026082102)
    parser.add_argument("--output-root", type=Path, default=ROOT / "log" / "v1_stage2_probe_sweep")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, ckpt = load_frozen_model(args.checkpoint, device, dtype)
    data_cfg = make_dataclass(FinalStage2DataConfig, ckpt["data_config"])
    data_cfg.batch_size = args.batch_size
    data_cfg.dataset_weights = args.dataset_weights
    data_cfg.history_iw_chunks = args.history_iw_chunks
    data_cfg.empty_hist_prob = 0.0
    data_cfg.empty_cur_prob = 0.0
    data_cfg.seed = args.seed
    train_builder = FinalStage2BatchBuilder(data_cfg)
    eval_cfg = make_dataclass(FinalStage2DataConfig, ckpt["data_config"])
    eval_cfg.batch_size = args.batch_size
    eval_cfg.dataset_weights = args.dataset_weights
    eval_cfg.history_iw_chunks = args.history_iw_chunks
    eval_cfg.empty_hist_prob = 0.0
    eval_cfg.empty_cur_prob = 0.0
    eval_cfg.seed = args.seed + 10000
    eval_builder = FinalStage2BatchBuilder(eval_cfg)

    probes = nn.ModuleDict(
        {
            str(layer): DenseCameraQueryPoseProbe(hidden_dim=model.cfg.hidden_dim, num_heads=8).to(device=device)
            for layer in layers
        }
    )
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    capture = LayerCapture(model, layers)

    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    config = {
        "event": "stage2_probe_sweep_start",
        "time": now(),
        "checkpoint": str(args.checkpoint),
        "train_step": int(ckpt["step"]),
        "run_dir": str(run_dir),
        "layers": layers,
        "probe": "DenseCameraQueryPoseProbe(camera query cross-attention over per-frame spatial tokens)",
        "frozen": "FinalStage2WanModel checkpoint frozen; only probe heads train",
        "data_config": data_cfg.to_dict(),
        "args": vars(args) | {"checkpoint": str(args.checkpoint), "output_root": str(args.output_root)},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    train_log = (run_dir / "train.jsonl").open("a")
    best: dict[str, Any] = {"layer": None, "pose_mse": float("inf"), "step": 0}
    started = time.time()
    try:
        for step in range(1, args.steps + 1):
            batch, meta = train_builder.next_batch(device=device, dtype=dtype)
            capture.clear()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                _ = model.forward_core(batch, return_prefix_hidden=False)
            target = batch["pose_target"].to(device=device, dtype=torch.float32)
            mask = batch["pose_mask"].to(device=device)
            latent_shape = (batch["z_obs"].shape[2], batch["z_obs"].shape[3], batch["z_obs"].shape[4])
            losses: dict[int, torch.Tensor] = {}
            metrics: dict[int, dict[str, float]] = {}
            for layer in layers:
                obs_hidden = model.current_obs_hidden(capture.outputs[layer], batch["z_obs"])
                pred = probes[str(layer)](obs_hidden, latent_shape=latent_shape)
                loss = masked_pose_mse(pred, target, mask)
                losses[layer] = loss
                metrics[layer] = pose_metrics(pred.detach(), target, mask)
            total_loss = torch.stack(list(losses.values())).mean()
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(probes.parameters(), args.grad_clip)
            optimizer.step()

            row = {
                "event": "train",
                "time": now(),
                "step": step,
                "seconds_total": time.time() - started,
                "datasets": meta.get("datasets"),
                "history_iw": meta.get("history_iw"),
                "loss_mean": float(total_loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "layers": {
                    str(layer): {"loss": float(losses[layer].detach().cpu().item()), **metrics[layer]}
                    for layer in layers
                },
            }
            train_log.write(json.dumps(row, ensure_ascii=False) + "\n")
            train_log.flush()
            writer.add_scalar("train/loss_mean", row["loss_mean"], step)
            writer.add_scalar("train/grad_norm", row["grad_norm"], step)
            for layer in layers:
                writer.add_scalar(f"train_layer{layer}/pose_mse", metrics[layer]["pose_mse"], step)
                writer.add_scalar(f"train_layer{layer}/translation_mae", metrics[layer]["translation_mae"], step)
                writer.add_scalar(f"train_layer{layer}/rotation_angle_deg", metrics[layer]["rotation_angle_deg"], step)
                writer.add_scalar(f"train_layer{layer}/fov_mae", metrics[layer]["fov_mae"], step)

            if step == 1 or (args.eval_every > 0 and step % args.eval_every == 0) or step == args.steps:
                eval_rows = evaluate_probes(
                    model=model,
                    capture=capture,
                    probes=probes,
                    builder=eval_builder,
                    device=device,
                    dtype=dtype,
                    layers=layers,
                    batches=args.eval_batches,
                )
                eval_payload = {"event": "eval", "time": now(), "step": step, "layers": {str(k): v for k, v in eval_rows.items()}}
                (run_dir / f"eval_step_{step:06d}.json").write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False) + "\n")
                train_log.write(json.dumps(eval_payload, ensure_ascii=False) + "\n")
                train_log.flush()
                for layer, vals in eval_rows.items():
                    for key, value in vals.items():
                        writer.add_scalar(f"eval_layer{layer}/{key}", value, step)
                    if vals["pose_mse"] < best["pose_mse"]:
                        best = {"layer": layer, "pose_mse": vals["pose_mse"], "step": step, "metrics": vals}
                writer.flush()

        torch.save(
            {
                "step": args.steps,
                "checkpoint": str(args.checkpoint),
                "train_step": int(ckpt["step"]),
                "layers": layers,
                "probes": probes.state_dict(),
                "best": best,
                "config": config,
            },
            run_dir / "probe_heads_final.pt",
        )
        (run_dir / "summary.json").write_text(
            json.dumps({"event": "stage2_probe_sweep_complete", "time": now(), "best": best, "run_dir": str(run_dir)}, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps({"status": "ok", "run_dir": str(run_dir), "best": best}, indent=2, ensure_ascii=False))
    finally:
        capture.close()
        train_log.close()
        writer.close()


if __name__ == "__main__":
    main()
