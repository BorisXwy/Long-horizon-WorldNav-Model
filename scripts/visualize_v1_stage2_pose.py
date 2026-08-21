#!/usr/bin/env python3
"""Visualize Stage2 pose GT vs prediction for final-standard checkpoints.

This script intentionally does not decode RGB/video.  It runs the full model
forward path, saves pose tensors, and writes lightweight trajectory/error
plots so it can be used while training occupies most GPU memory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from eval_v1_stage2_final_cotrain import load_model, make_dataclass, masked_mean, torch_dtype  # noqa: E402
from nav.v1.stage2_final import FinalStage2BatchBuilder, FinalStage2DataConfig  # noqa: E402
from train_v1_stage2_probe_sweep import DenseCameraQueryPoseProbe, LayerCapture  # noqa: E402


def now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def pose_errors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    valid = mask.bool()
    pred_v = pred[valid].float()
    target_v = target[valid].float()
    if pred_v.numel() == 0:
        return {"num_valid": 0}
    trans_l2 = torch.linalg.norm(pred_v[:, :3] - target_v[:, :3], dim=-1)
    trans_abs = (pred_v[:, :3] - target_v[:, :3]).abs()
    rot_deg = []
    for p, t in zip(pred_v[:, 3:7], target_v[:, 3:7]):
        p = p / p.norm().clamp_min(1e-8)
        t = t / t.norm().clamp_min(1e-8)
        dot = torch.abs((p * t).sum()).clamp(max=1.0)
        rot_deg.append(float((2.0 * torch.acos(dot) * 180.0 / torch.pi).cpu()))
    fov_abs = (pred_v[:, 7:9] - target_v[:, 7:9]).abs()
    return {
        "num_valid": int(valid.sum().item()),
        "translation_l2_per_frame": trans_l2.cpu().tolist(),
        "translation_l2_mean": float(trans_l2.mean().item()),
        "translation_abs_xyz_mean": trans_abs.mean(dim=0).cpu().tolist(),
        "rotation_angle_deg_per_frame": rot_deg,
        "rotation_angle_deg_mean": float(np.mean(rot_deg)),
        "fov_abs_per_frame": fov_abs.cpu().tolist(),
        "fov_abs_mean": float(fov_abs.mean().item()),
    }


def _valid_np(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = mask.bool().cpu().numpy()
    return pred.float().cpu().numpy()[valid], target.float().cpu().numpy()[valid], np.nonzero(valid)[0]


def plot_trajectory(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, out: Path, title: str) -> None:
    pred_np, target_np, frame_ids = _valid_np(pred, target, mask)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if len(frame_ids) > 0:
        ax.plot(target_np[:, 0], target_np[:, 1], target_np[:, 2], "o-", label="GT", linewidth=2)
        ax.plot(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], "x--", label="Pred", linewidth=2)
        for i, fidx in enumerate(frame_ids):
            ax.text(target_np[i, 0], target_np[i, 1], target_np[i, 2], str(int(fidx)), fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_trajectory_compare(
    preds: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    out: Path,
    title: str,
) -> None:
    target_valid = target.float().cpu().numpy()[mask.bool().cpu().numpy()]
    frame_ids = np.nonzero(mask.bool().cpu().numpy())[0]
    pred_valid = {
        name: pred.float().cpu().numpy()[mask.bool().cpu().numpy()]
        for name, pred in preds.items()
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []
    if len(target_valid) > 0:
        all_points.append(target_valid[:, :3])
        ax.plot(target_valid[:, 0], target_valid[:, 1], target_valid[:, 2], "o-", label="GT", linewidth=2.5)
        for i, fidx in enumerate(frame_ids):
            ax.text(target_valid[i, 0], target_valid[i, 1], target_valid[i, 2], str(int(fidx)), fontsize=8)
    markers = ["x--", "s--", "^--", "d--"]
    for index, (name, pred_np) in enumerate(pred_valid.items()):
        if len(pred_np) == 0:
            continue
        all_points.append(pred_np[:, :3])
        ax.plot(
            pred_np[:, 0],
            pred_np[:, 1],
            pred_np[:, 2],
            markers[index % len(markers)],
            label=name,
            linewidth=2,
        )
    if all_points:
        pts = np.concatenate(all_points, axis=0)
        center = pts.mean(axis=0)
        radius = max(float(np.max(np.abs(pts - center))), 1e-3)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_projections(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, out: Path, title: str) -> None:
    pred_np, target_np, _ = _valid_np(pred, target, mask)
    pairs = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (a, b, ia, ib) in zip(axes, pairs):
        if len(pred_np) > 0:
            ax.plot(target_np[:, ia], target_np[:, ib], "o-", label="GT")
            ax.plot(pred_np[:, ia], pred_np[:, ib], "x--", label="Pred")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_errors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, out: Path, title: str) -> None:
    pred_np, target_np, frame_ids = _valid_np(pred, target, mask)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    if len(pred_np) > 0:
        trans_l2 = np.linalg.norm(pred_np[:, :3] - target_np[:, :3], axis=-1)
        pred_q = pred_np[:, 3:7]
        target_q = target_np[:, 3:7]
        pred_q = pred_q / np.clip(np.linalg.norm(pred_q, axis=-1, keepdims=True), 1e-8, None)
        target_q = target_q / np.clip(np.linalg.norm(target_q, axis=-1, keepdims=True), 1e-8, None)
        rot_deg = 2.0 * np.arccos(np.clip(np.abs((pred_q * target_q).sum(axis=-1)), 0.0, 1.0)) * 180.0 / math.pi
        fov_l1 = np.abs(pred_np[:, 7:9] - target_np[:, 7:9]).mean(axis=-1)
        axes[0].plot(frame_ids, trans_l2, "o-", color="tab:blue")
        axes[1].plot(frame_ids, rot_deg, "o-", color="tab:orange")
        axes[2].plot(frame_ids, fov_l1, "o-", color="tab:green")
    axes[0].set_ylabel("translation L2")
    axes[1].set_ylabel("rotation deg")
    axes[2].set_ylabel("FOV L1")
    axes[2].set_xlabel("frame index in current chunk")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_components(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, out: Path, title: str) -> None:
    pred_np, target_np, frame_ids = _valid_np(pred, target, mask)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    names = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        if len(pred_np) > 0:
            ax.plot(frame_ids, target_np[:, i], "o-", label=f"GT {names[i]}")
            ax.plot(frame_ids, pred_np[:, i], "x--", label=f"Pred {names[i]}")
        ax.set_ylabel(names[i])
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("frame index in current chunk")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def load_probe_head(
    *,
    probe_checkpoint: Path,
    layer: int,
    hidden_dim: int,
    device: torch.device,
) -> DenseCameraQueryPoseProbe:
    payload = torch.load(probe_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("probes")
    if not isinstance(state, dict):
        raise RuntimeError(f"{probe_checkpoint} does not contain a probe state dict under key 'probes'")
    prefix = f"{layer}."
    layer_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not layer_state:
        raise RuntimeError(f"{probe_checkpoint} does not contain layer {layer} probe weights")
    probe = DenseCameraQueryPoseProbe(hidden_dim=hidden_dim, num_heads=8).to(device=device)
    probe.load_state_dict(layer_state, strict=True)
    probe.eval()
    return probe


def save_pose_artifacts(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    meta: dict[str, Any],
    sample_dir: Path,
    head_name: str,
    title: str,
) -> dict[str, Any]:
    head_dir = sample_dir / head_name
    head_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "pose_pred": pred, "pose_target": target, "pose_mask": mask}, head_dir / "pose.pt")
    errs = pose_errors(pred, target, mask)
    (head_dir / "pose_values.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "pose_format": "[tx,ty,tz,qw,qx,qy,qz,fov_x,fov_y]",
                "translation_note": "前三维是当前 Stage2 监督目标中的 relative transform translation，不是额外换算后的 absolute camera center。",
                "pose_pred": pred.tolist(),
                "pose_target": target.tolist(),
                "pose_mask": mask.tolist(),
                "errors": errs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    plot_trajectory(pred, target, mask, head_dir / "pose_trajectory_xyz.png", title)
    plot_projections(pred, target, mask, head_dir / "pose_projections.png", title)
    plot_errors(pred, target, mask, head_dir / "pose_errors.png", title)
    plot_components(pred, target, mask, head_dir / "pose_translation_components.png", title)
    return {"head": head_name, "head_dir": str(head_dir), "errors": errs}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--lambda-pose", type=float, default=0.1)
    parser.add_argument("--dataset-weights", default="re10k=1.0")
    parser.add_argument("--history-iw-chunks", default="1")
    parser.add_argument("--eval-seed", type=int, default=20260820)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output-root", type=Path, default=ROOT / "result" / "v1_stage2_pose_visualization")
    parser.add_argument("--probe-checkpoint", type=Path, default=None)
    parser.add_argument("--probe-layer", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)

    model, ckpt = load_model(args.checkpoint, device, dtype)
    probe = None
    capture = None
    if args.probe_checkpoint is not None:
        probe = load_probe_head(
            probe_checkpoint=args.probe_checkpoint,
            layer=args.probe_layer,
            hidden_dim=model.cfg.hidden_dim,
            device=device,
        )
        capture = LayerCapture(model, [args.probe_layer])
    data_cfg = make_dataclass(FinalStage2DataConfig, ckpt["data_config"])
    data_cfg.batch_size = args.batch_size
    data_cfg.seed = args.eval_seed
    data_cfg.dataset_weights = args.dataset_weights
    data_cfg.history_iw_chunks = args.history_iw_chunks
    data_cfg.empty_hist_prob = 0.0
    data_cfg.empty_cur_prob = 0.0
    builder = FinalStage2BatchBuilder(data_cfg)

    run_name = args.run_name or f"{args.checkpoint.stem}_pose_vis_{now_compact()}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "train_step": int(ckpt["step"]),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "dataset_weights": args.dataset_weights,
        "history_iw_chunks": args.history_iw_chunks,
        "samples": [],
        "pose_format": "[tx,ty,tz,qw,qx,qy,qz,fov_x,fov_y]",
        "trajectory_note": "轨迹图画的是当前 Stage2 监督目标中的 relative transform translation 三维分量；它用于检查 pred/GT 对齐，不额外假设其为全局 absolute camera center。",
        "probe_checkpoint": str(args.probe_checkpoint) if args.probe_checkpoint is not None else None,
        "probe_layer": args.probe_layer if args.probe_checkpoint is not None else None,
    }

    totals: dict[str, dict[str, list[float]]] = {}
    for idx in range(args.num_samples):
        batch, meta = builder.next_batch(device=device, dtype=dtype)
        if capture is not None:
            capture.clear()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            out = model.forward_stage2(batch, lambda_pose=args.lambda_pose)
        pose_target = batch["pose_target"][0].detach().float().cpu()
        pose_mask = batch["pose_mask"][0].detach().float().cpu()
        sample_dir = output_dir / "samples" / f"{idx:03d}_{','.join(meta.get('datasets') or ['unknown'])}_histIW{meta.get('history_iw', 'x')}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        plot_title = f"sample {idx:03d} | {','.join(meta.get('datasets') or ['unknown'])} | hist IW{meta.get('history_iw', 'x')}"

        heads: dict[str, torch.Tensor] = {
            "attached_head": out["pose_pred"][0].detach().float().cpu(),
        }
        if probe is not None and capture is not None:
            latent_shape = (batch["z_obs"].shape[2], batch["z_obs"].shape[3], batch["z_obs"].shape[4])
            obs_hidden = model.current_obs_hidden(capture.outputs[args.probe_layer], batch["z_obs"])
            probe_pred = probe(obs_hidden.float(), latent_shape=latent_shape)[0].detach().float().cpu()
            heads[f"probe_layer{args.probe_layer}"] = probe_pred

        sample_rows = []
        for head_name, pose_pred in heads.items():
            row = save_pose_artifacts(
                pred=pose_pred,
                target=pose_target,
                mask=pose_mask,
                meta=meta,
                sample_dir=sample_dir,
                head_name=head_name,
                title=f"{plot_title} | {head_name}",
            )
            errs = row["errors"]
            head_totals = totals.setdefault(head_name, {"translation_l2": [], "rotation_deg": [], "fov_abs": [], "pose_mse": []})
            if errs.get("num_valid", 0) > 0:
                head_totals["translation_l2"].append(errs["translation_l2_mean"])
                head_totals["rotation_deg"].append(errs["rotation_angle_deg_mean"])
                head_totals["fov_abs"].append(errs["fov_abs_mean"])
                head_totals["pose_mse"].append(float(masked_mean((pose_pred - pose_target).pow(2), pose_mask).item()))
            sample_rows.append(row)
        if len(heads) > 1:
            plot_trajectory_compare(
                heads,
                pose_target,
                pose_mask,
                sample_dir / "pose_trajectory_compare_xyz.png",
                f"{plot_title} | shared-axis comparison",
            )
        summary["samples"].append({"sample_dir": str(sample_dir), "meta": meta, "heads": sample_rows})
        print(json.dumps({"event": "pose_visualized", "sample": idx, "sample_dir": str(sample_dir), "heads": sample_rows}, ensure_ascii=False), flush=True)

    summary["aggregate"] = {
        head_name: {
            key: (float(np.mean(values)) if values else None)
            for key, values in head_totals.items()
        }
        for head_name, head_totals in totals.items()
    }
    (output_dir / "pose_visualization_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "aggregate": summary["aggregate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
