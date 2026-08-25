#!/usr/bin/env python3
"""Full multi-step denoising sampler for final-standard Stage2 checkpoints."""

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
    model.remove_hmpc()
    result = model.load_state_dict(ckpt["model"], strict=False)
    bad_missing = [key for key in result.missing_keys if not key.startswith("backbone.latent_encoder.")]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing[:20]} unexpected={result.unexpected_keys[:20]}")
    return model.to(device=device, dtype=dtype).eval(), ckpt


def stage2_data_payload(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Read the shared video/3D replay config from Stage2 or Stage3 checkpoints."""

    payload = ckpt.get("data_config")
    if payload is None:
        payload = ckpt.get("stage2_replay_data_config")
    if payload is None:
        raise RuntimeError("checkpoint has neither data_config nor stage2_replay_data_config")
    return payload


def rflow_timesteps(steps: int, *, shift: float, device: torch.device) -> torch.Tensor:
    raw = torch.linspace(1.0, 0.001, int(steps), device=device, dtype=torch.float32)
    return shift * raw / (1.0 + (shift - 1.0) * raw)


def estimate_x0_from_velocity(batch: dict[str, torch.Tensor], velocity: torch.Tensor) -> torch.Tensor:
    t = batch["visual_timestep"].to(device=velocity.device, dtype=velocity.dtype)
    while t.ndim < velocity.ndim:
        t = t.view(*t.shape, *([1] * (velocity.ndim - t.ndim)))
    return batch["z_future_noisy"].to(device=velocity.device, dtype=velocity.dtype) - t * velocity


@torch.no_grad()
def sample_rflow(
    model: FinalStage2WanModel,
    batch: dict[str, torch.Tensor],
    *,
    steps: int,
    shift: float,
    device: torch.device,
    dtype: torch.dtype,
    action_noise_mode: str,
) -> tuple[torch.Tensor, list[float], list[float]]:
    z = torch.randn_like(batch["z_future_target"], dtype=dtype, device=device)
    if action_noise_mode == "zero":
        batch = dict(batch)
        batch["a_noise"] = torch.zeros_like(batch["a_noise"])
        batch["action_timestep"] = torch.zeros_like(batch["action_timestep"])
    timesteps = rflow_timesteps(steps, shift=shift, device=device)
    forward_times: list[float] = []
    t_values: list[float] = []
    for index, timestep in enumerate(timesteps):
        cur_batch = dict(batch)
        cur_batch["z_future_noisy"] = z
        cur_batch["visual_timestep"] = timestep.expand(z.shape[0]).to(device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            out = model.forward_core(cur_batch, return_prefix_hidden=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_times.append(time.perf_counter() - start)
        t_values.append(float(timestep.detach().cpu().item()))
        pred_velocity = out["future_velocity"].to(dtype)
        next_t = timesteps[index + 1] if index < len(timesteps) - 1 else timesteps.new_zeros(())
        dt = (timestep - next_t).to(device=device, dtype=dtype)
        z = z - pred_velocity * dt.view(1, *([1] * (z.ndim - 1)))
    return z, forward_times, t_values


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


@torch.no_grad()
def decode_latents(vae: Any, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    return vae.decode(latents.to(device=device, dtype=torch.bfloat16)).float().cpu()


def tensor_to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    video = ((video.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous().cpu()


def save_video(video: torch.Tensor, path: Path, fps: int = 12) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = tensor_to_uint8_video(video).numpy()
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def rgb_metrics(pred_px: torch.Tensor, target_px: torch.Tensor) -> dict[str, float]:
    pred = ((pred_px + 1.0) * 0.5).clamp(0, 1)
    target = ((target_px + 1.0) * 0.5).clamp(0, 1)
    mse = F.mse_loss(pred, target)
    return {
        "rgb_mse": float(mse.item()),
        "rgb_l1": float((pred - target).abs().mean().item()),
        "rgb_psnr_db": float((-10.0 * torch.log10(mse.clamp_min(1e-8))).item()),
        "rgb_pred_mean": float(pred.mean().item()),
        "rgb_pred_std": float(pred.std(unbiased=False).item()),
        "rgb_target_mean": float(target.mean().item()),
        "rgb_target_std": float(target.std(unbiased=False).item()),
    }


def latent_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(pred.float(), target.float())
    return {
        "latent_mse": float(mse.item()),
        "latent_l1": float((pred.float() - target.float()).abs().mean().item()),
        "latent_cosine": float(F.cosine_similarity(pred.float().flatten(1), target.float().flatten(1), dim=1).mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", default=f"full_denoise_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--shift", type=float, default=7.0)
    parser.add_argument("--dataset-weights", default="re10k=0.20,spatialvid=0.45,dl3dv=0.35")
    parser.add_argument("--history-iw-chunks", default="1,4,8,16")
    parser.add_argument("--eval-seed", type=int, default=20260819)
    parser.add_argument("--action-noise-mode", choices=("random", "zero"), default="random")
    parser.add_argument("--output-root", type=Path, default=ROOT / "result" / "v1_stage2_final_denoise")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)
    model, ckpt = load_model(args.checkpoint, device, dtype)
    data_cfg = make_dataclass(FinalStage2DataConfig, stage2_data_payload(ckpt))
    data_cfg.batch_size = 1
    data_cfg.seed = args.eval_seed
    data_cfg.empty_hist_prob = 0.0
    data_cfg.empty_cur_prob = 0.0
    data_cfg.dataset_weights = args.dataset_weights
    data_cfg.history_iw_chunks = args.history_iw_chunks
    builder = FinalStage2BatchBuilder(data_cfg)
    vae = load_wan_vae(device)
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for index in range(args.num_samples):
        batch, meta = builder.next_batch(device=device, dtype=dtype)
        # Denoise diagnostic only: this uses the eval batch's z_noisy, which is
        # constructed from the GT target.  It must not be reported as generation
        # quality.  True generation is the full RFlow chain below.
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            proxy_out = model.forward_core(batch, return_prefix_hidden=False)
        proxy_latent = estimate_x0_from_velocity(batch, proxy_out["future_velocity"]).float()
        denoised, forward_times, t_values = sample_rflow(
            model,
            batch,
            steps=args.steps,
            shift=args.shift,
            device=device,
            dtype=dtype,
            action_noise_mode=args.action_noise_mode,
        )
        target = batch["z_future_target"].float()
        obs = batch["z_obs"].float()
        denoised_px = decode_latents(vae, denoised.float(), device)
        proxy_px = decode_latents(vae, proxy_latent, device)
        target_px = decode_latents(vae, target, device)
        obs_px = decode_latents(vae, obs, device)
        sample_dir = output_dir / "samples" / f"{index:03d}_{','.join(meta.get('datasets') or ['unknown'])}_histIW{meta.get('history_iw')}"
        save_video(obs_px[0], sample_dir / "obs.mp4")
        save_video(target_px[0], sample_dir / "gt_future.mp4")
        save_video(proxy_px[0], sample_dir / "denoise_diagnostic_gt_mixed_one_step.mp4")
        save_video(denoised_px[0], sample_dir / f"full_denoise_{args.steps}steps.mp4")
        sample_report = {
            "sample_index": index,
            "sample_dir": str(sample_dir),
            "meta": meta,
            "steps": args.steps,
            "shift": args.shift,
            "action_noise_mode": args.action_noise_mode,
            "denoise_forward_times": forward_times,
            "denoise_forward_time_mean": sum(forward_times) / max(len(forward_times), 1),
            "denoise_forward_time_total": sum(forward_times),
            "timesteps": t_values,
            "denoise_diagnostic_gt_mixed_one_step": {
                "note": "Not generation: z_noisy is mixed from GT target and random noise.",
                **latent_metrics(proxy_latent, target),
                **rgb_metrics(proxy_px, target_px),
            },
            "full_denoise": {
                **latent_metrics(denoised.float(), target),
                **rgb_metrics(denoised_px, target_px),
            },
        }
        (sample_dir / "metrics.json").write_text(json.dumps(sample_report, indent=2, ensure_ascii=False) + "\n")
        samples.append(sample_report)
        print(json.dumps({"event": "sample_done", "index": index, "sample_dir": str(sample_dir), "full_rgb_psnr": sample_report["full_denoise"]["rgb_psnr_db"]}, ensure_ascii=False), flush=True)

    def mean_metric(section: str, key: str) -> float:
        return sum(float(s[section][key]) for s in samples) / max(len(samples), 1)

    summary = {
        "event": "stage2_final_full_denoise",
        "checkpoint": str(args.checkpoint),
        "train_step": int(ckpt["step"]),
        "run_name": args.run_name,
        "output_dir": str(output_dir),
        "num_samples": args.num_samples,
        "steps": args.steps,
        "shift": args.shift,
        "eval_seed": args.eval_seed,
        "dataset_weights": args.dataset_weights,
        "history_iw_chunks": args.history_iw_chunks,
        "action_noise_mode": args.action_noise_mode,
        "seconds_total": time.perf_counter() - start_all,
        "mean": {
            "denoise_diagnostic_gt_mixed_one_step_rgb_psnr_db": mean_metric("denoise_diagnostic_gt_mixed_one_step", "rgb_psnr_db"),
            "full_denoise_rgb_psnr_db": mean_metric("full_denoise", "rgb_psnr_db"),
            "denoise_diagnostic_gt_mixed_one_step_rgb_l1": mean_metric("denoise_diagnostic_gt_mixed_one_step", "rgb_l1"),
            "full_denoise_rgb_l1": mean_metric("full_denoise", "rgb_l1"),
            "denoise_diagnostic_gt_mixed_one_step_latent_mse": mean_metric("denoise_diagnostic_gt_mixed_one_step", "latent_mse"),
            "full_denoise_latent_mse": mean_metric("full_denoise", "latent_mse"),
            "denoise_forward_time_total": sum(float(s["denoise_forward_time_total"]) for s in samples) / max(len(samples), 1),
        },
        "samples": samples,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
