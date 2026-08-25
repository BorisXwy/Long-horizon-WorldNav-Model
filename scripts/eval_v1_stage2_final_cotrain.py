#!/usr/bin/env python3
"""Evaluate final-standard V1 Stage2 cotrain checkpoints.

This evaluator matches ``scripts/train_v1_stage2_final_cotrain.py`` and uses
the complete final model path:

* full ``FinalStage2WanModel`` checkpoint;
* full Register rollout with the checkpoint's final-standard config;
* ``Z_obs + A_noise + Z_future_noise`` main stream;
* ``Register + text + A_cur`` condition stream for the current run;
* visual RFlow/velocity metrics, pose metrics, and optional Wan VAE decode.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
INF_WORLD_ROOT = Path(os.environ.get("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World")))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(INF_WORLD_ROOT))

from nav.v1.stage2_final import FinalStage2BatchBuilder, FinalStage2DataConfig, FinalStage2WanConfig, FinalStage2WanModel  # noqa: E402


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


def load_model(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> tuple[FinalStage2WanModel, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = make_dataclass(FinalStage2WanConfig, ckpt["model_config"])
    model = FinalStage2WanModel(cfg)
    # Training checkpoints are saved after ``load_wan_checkpoint()``, which
    # replaces HPMC/latent_encoder with Identity.  Recreate that runtime state
    # before loading model weights.
    model.remove_hmpc()
    result = model.load_state_dict(ckpt["model"], strict=False)
    bad_missing = [key for key in result.missing_keys if not key.startswith("backbone.latent_encoder.")]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            "Stage2 final checkpoint mismatch: "
            f"missing={bad_missing[:20]} unexpected={result.unexpected_keys[:20]}"
        )
    return model.to(device=device, dtype=dtype).eval(), ckpt


def stage2_data_payload(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Read the shared video/3D replay config from Stage2 or Stage3 checkpoints."""

    payload = ckpt.get("data_config")
    if payload is None:
        payload = ckpt.get("stage2_replay_data_config")
    if payload is None:
        raise RuntimeError("checkpoint has neither data_config nor stage2_replay_data_config")
    return payload


def load_wan_vae(device: torch.device) -> Any:
    from omegaconf import OmegaConf
    from infworld.utils.prepare_dataloader import get_obj_from_str

    config = OmegaConf.load(INF_WORLD_ROOT / "configs" / "infworld_config.yaml")
    vae_pth = Path(str(config.vae_cfg.vae_pth))
    if not vae_pth.is_absolute():
        public_vae = Path("/sharedata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
        config.vae_cfg.vae_pth = str(public_vae if public_vae.exists() else INF_WORLD_ROOT / vae_pth)
    vae = get_obj_from_str(config.vae_target)(**config.vae_cfg).to(device)
    vae.eval()
    return vae


def tensor_to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f"expected [C,T,H,W], got {tuple(video.shape)}")
    video = ((video.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous().cpu()


def save_video(video: torch.Tensor, path: Path, fps: int = 12) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = tensor_to_uint8_video(video).numpy()
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


@torch.no_grad()
def decode_latents(vae: Any, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    return vae.decode(latents.to(device=device, dtype=torch.bfloat16)).float().cpu()


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)


def quat_angle_deg(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (pred * target).sum(dim=-1).abs().clamp(max=1.0)
    angle = 2.0 * torch.acos(dot) * 180.0 / torch.pi
    return (angle * mask.to(angle.dtype)).sum() / mask.sum().clamp_min(1.0)


def estimate_x0_from_velocity(batch: dict[str, torch.Tensor], velocity: torch.Tensor) -> torch.Tensor:
    t = batch["visual_timestep"].to(device=velocity.device, dtype=velocity.dtype)
    while t.ndim < velocity.ndim:
        t = t.view(*t.shape, *([1] * (velocity.ndim - t.ndim)))
    return batch["z_future_noisy"].to(device=velocity.device, dtype=velocity.dtype) - t * velocity


def metric_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


@torch.no_grad()
def evaluate(
    *,
    model: FinalStage2WanModel,
    builder: FinalStage2BatchBuilder,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
    lambda_pose: float,
    decode_rgb: bool,
    decode_batches: int,
    save_videos: int,
    output_dir: Path,
) -> dict[str, Any]:
    vae = load_wan_vae(device) if decode_rgb else None
    sums: dict[str, float] = {}
    by_history: dict[str, dict[str, float]] = {}
    by_dataset: dict[str, dict[str, float]] = {}
    rgb_sums: dict[str, float] = {}
    rgb_count = 0
    total_pose_valid = 0.0
    count = 0
    started = time.time()
    for index in range(max_batches):
        batch, meta = builder.next_batch(device=device, dtype=dtype)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            out = model.forward_stage2(batch, lambda_pose=lambda_pose)
        velocity = out["future_velocity"]
        target_velocity = batch["z_future_noise"].to(velocity.dtype) - batch["z_future_target"].to(velocity.dtype)
        z_pred = estimate_x0_from_velocity(batch, velocity).float()
        z_target = batch["z_future_target"].float()
        z_noisy = batch["z_future_noisy"].float()
        latent_mse = F.mse_loss(z_pred, z_target)
        baseline_mse = F.mse_loss(z_noisy, z_target)
        flat_pred = z_pred.flatten(1)
        flat_target = z_target.flatten(1)
        pose_pred = out["pose_pred"].float()
        pose_target = batch["pose_target"].float()
        pose_mask = batch["pose_mask"].float()
        pose_valid = float(pose_mask.sum().detach().cpu().item())
        metrics = {
            "loss_total": out["loss"],
            "loss_visual_velocity_mse": out["loss_visual"],
            "velocity_target_mse": F.mse_loss(velocity.float(), target_velocity.float()),
            "latent_x0_mse": latent_mse,
            "latent_x0_rmse": torch.sqrt(latent_mse),
            "latent_x0_l1": (z_pred - z_target).abs().mean(),
            "baseline_noisy_latent_mse": baseline_mse,
            "latent_mse_improvement_vs_noisy": baseline_mse - latent_mse,
            "latent_x0_cosine": F.cosine_similarity(flat_pred, flat_target, dim=1).mean(),
            "loss_pose_mse": out["loss_pose"],
            "pose_translation_mae": masked_mean((pose_pred[..., :3] - pose_target[..., :3]).abs(), pose_mask),
            "pose_rotation_angle_deg": quat_angle_deg(pose_pred[..., 3:7], pose_target[..., 3:7], pose_mask),
            "pose_fov_mae": masked_mean((pose_pred[..., 7:9] - pose_target[..., 7:9]).abs(), pose_mask),
            "pose_valid_frames": pose_valid,
        }
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + metric_value(value)
        hkey = f"IW{meta['history_iw']}"
        dkey = ",".join(meta.get("datasets") or ["unknown"])
        for table, key in ((by_history, hkey), (by_dataset, dkey)):
            row = table.setdefault(key, {"n": 0.0})
            row["n"] += 1.0
            for metric_name in ("loss_visual_velocity_mse", "latent_x0_mse", "baseline_noisy_latent_mse", "loss_pose_mse", "pose_valid_frames"):
                row[metric_name] = row.get(metric_name, 0.0) + metric_value(metrics[metric_name])
        total_pose_valid += pose_valid

        if vae is not None and (decode_batches <= 0 or rgb_count < decode_batches):
            pred_px = decode_latents(vae, z_pred, device)
            target_px = decode_latents(vae, z_target, device)
            obs_px = decode_latents(vae, batch["z_obs"], device)
            pred_01 = ((pred_px + 1.0) * 0.5).clamp(0, 1)
            target_01 = ((target_px + 1.0) * 0.5).clamp(0, 1)
            rgb_mse = F.mse_loss(pred_01, target_01)
            rgb_metrics = {
                "rgb_mse": rgb_mse,
                "rgb_l1": (pred_01 - target_01).abs().mean(),
                "rgb_psnr_db": -10.0 * torch.log10(rgb_mse.clamp_min(1e-8)),
                "rgb_pred_mean": pred_01.mean(),
                "rgb_pred_std": pred_01.std(unbiased=False),
                "rgb_target_mean": target_01.mean(),
                "rgb_target_std": target_01.std(unbiased=False),
            }
            for key, value in rgb_metrics.items():
                rgb_sums[key] = rgb_sums.get(key, 0.0) + metric_value(value)
            if rgb_count < save_videos:
                sample_dir = output_dir / "samples" / f"{rgb_count:03d}_{dkey}_hist{hkey}"
                save_video(obs_px[0], sample_dir / "obs.mp4")
                save_video(target_px[0], sample_dir / "gt_future.mp4")
                save_video(pred_px[0], sample_dir / "pred_future_proxy.mp4")
                torch.save(
                    {
                        "meta": meta,
                        "pose_pred": pose_pred[0].detach().cpu(),
                        "pose_target": pose_target[0].detach().cpu(),
                        "pose_mask": pose_mask[0].detach().cpu(),
                    },
                    sample_dir / "sample_meta.pt",
                )
            rgb_count += 1
        count += 1
        if (index + 1) % 10 == 0:
            print(json.dumps({"event": "progress", "time": now(), "batches": index + 1, "pose_valid_frames": total_pose_valid}, ensure_ascii=False), flush=True)

    def average_table(table: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, row in table.items():
            n = max(row.get("n", 0.0), 1.0)
            out[key] = {metric: (value / n if metric != "n" else value) for metric, value in row.items()}
        return out

    metrics_out = {f"eval/{key}": value / max(count, 1) for key, value in sums.items()}
    metrics_out.update({f"eval_decode/{key}": value / max(rgb_count, 1) for key, value in rgb_sums.items()})
    metrics_out["eval/num_batches"] = float(count)
    metrics_out["eval_decode/num_batches"] = float(rgb_count)
    metrics_out["eval/seconds_total"] = time.time() - started
    metrics_out["eval/seconds_per_batch"] = metrics_out["eval/seconds_total"] / max(count, 1)
    return {
        "metrics": metrics_out,
        "by_history": average_table(by_history),
        "by_dataset": average_table(by_dataset),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--lambda-pose", type=float, default=0.1)
    parser.add_argument("--decode-rgb", action="store_true")
    parser.add_argument("--decode-batches", type=int, default=4)
    parser.add_argument("--save-videos", type=int, default=4)
    parser.add_argument("--dataset-weights", default="")
    parser.add_argument("--history-iw-chunks", default="")
    parser.add_argument("--eval-seed", type=int, default=20260819)
    parser.add_argument("--output-root", type=Path, default=ROOT / "result" / "v1_stage2_final_cotrain")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)
    model, ckpt = load_model(args.checkpoint, device, dtype)
    # Stage2 and Stage3 instantiate different post-backbone action heads, which
    # consume different amounts of RNG during construction.  Reset after model
    # loading so paired checkpoint evaluations use identical sampled windows,
    # timesteps, and diffusion noise.
    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)
    data_cfg = make_dataclass(FinalStage2DataConfig, stage2_data_payload(ckpt))
    data_cfg.batch_size = args.batch_size
    data_cfg.seed = args.eval_seed
    data_cfg.empty_hist_prob = 0.0
    data_cfg.empty_cur_prob = 0.0
    if args.dataset_weights:
        data_cfg.dataset_weights = args.dataset_weights
    if args.history_iw_chunks:
        data_cfg.history_iw_chunks = args.history_iw_chunks
    builder = FinalStage2BatchBuilder(data_cfg)
    run_name = args.run_name or f"{args.checkpoint.parent.parent.name}_{args.checkpoint.stem}_eval"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(
        model=model,
        builder=builder,
        device=device,
        dtype=dtype,
        max_batches=args.max_batches,
        lambda_pose=args.lambda_pose,
        decode_rgb=args.decode_rgb,
        decode_batches=args.decode_batches,
        save_videos=args.save_videos,
        output_dir=output_dir,
    )
    report = {
        "event": "stage2_final_eval",
        "time": now(),
        "checkpoint": str(args.checkpoint),
        "train_step": int(ckpt["step"]),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "eval_seed": args.eval_seed,
        "data_config": data_cfg.to_dict(),
        "model_config": ckpt["model_config"],
        "max_batches": args.max_batches,
        "decode_rgb": args.decode_rgb,
        "note": "生成指标使用 one-step RFlow x0 proxy: x0 = z_noisy - t * v_pred；不是完整多步 sampler。Pose 指标只在 pose_mask>0 的样本上有意义。",
        **result,
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", **report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
