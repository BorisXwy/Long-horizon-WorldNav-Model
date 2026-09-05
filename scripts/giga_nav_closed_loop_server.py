#!/usr/bin/env python3
"""Serve complete GigaNav + Wan VAE inference for Habitat closed-loop eval."""

from __future__ import annotations

import argparse
import hashlib
from multiprocessing.connection import Listener
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import center_crop


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "Infinite-World"))

from giga_nav import GigaNavConfig, GigaNavModel  # noqa: E402
from infworld.vae import WanVAEModelWrapper  # noqa: E402


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def preprocess_rgb(image: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Apply the same resize + center crop used by the T4 training cache."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB [H,W,3], got {image.shape} {image.dtype}")
    scale = max(height / image.shape[0], width / image.shape[1])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return center_crop(torch.from_numpy(resized.copy()).permute(2, 0, 1), [height, width])


def load_model(checkpoint: Path, device: torch.device) -> tuple[GigaNavModel, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_keys = set(GigaNavConfig.__dataclass_fields__)
    values = {key: value for key, value in payload.get("model_config", {}).items() if key in model_keys}
    if "backbone_checkpoint" in values:
        values["backbone_checkpoint"] = Path(values["backbone_checkpoint"])
    if "patch_size" in values:
        values["patch_size"] = tuple(values["patch_size"])
    model = GigaNavModel(GigaNavConfig(**values))
    model.load_state_dict(payload["model"], strict=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vae", type=Path, default=Path("/sharedata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=896)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, checkpoint_payload = load_model(args.checkpoint, device)
    vae = WanVAEModelWrapper(vae_pth=str(args.vae), dtype=dtype, device=str(device)).to(device)
    vae.eval().requires_grad_(False)

    text_embedding: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    current_episode: str | None = None
    inference_cache: dict[bytes, dict] = {}

    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    listener = Listener(str(args.socket), family="AF_UNIX")
    connection = listener.accept()
    data_config = checkpoint_payload.get("data_config", {})
    reference_delay = (
        0 if data_config.get("action_label_alignment") == "reference_frame" else 12
    )
    expected_window_frames = reference_delay + 1 if reference_delay else 1
    connection.send(
        {
            "status": "ready",
            "checkpoint": str(args.checkpoint),
            "structure": model.structural_report(),
            "training_reference_delay_steps": reference_delay,
            "online_vae_input_frames": expected_window_frames,
            "online_vae_encoded_frames": 1,
            "latent_padding": (
                "encode the rolling window's oldest frame as the causal reference and pad three zero planes"
                if expected_window_frames == 13
                else "causal T1 observation + three zero latent planes -> T4 model interface"
            ),
        }
    )
    try:
        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "close":
                connection.send({"status": "closed"})
                break
            if command == "reset":
                text_path = Path(request["text_embedding"])
                text_payload = torch.load(text_path, map_location="cpu", weights_only=False)
                text_embedding = text_payload["y"].to(device=device, dtype=dtype)
                text_mask = text_payload["y_mask"].to(device=device)
                current_episode = str(request["episode_id"])
                inference_cache.clear()
                connection.send({"status": "reset", "episode_id": current_episode, "text_embedding": str(text_path)})
                continue
            if command != "infer":
                raise ValueError(f"unknown command: {command!r}")
            if text_embedding is None or text_mask is None:
                raise RuntimeError("infer called before reset")

            start = time.perf_counter()
            rgb_window = np.asarray(request["rgb_window"], dtype=np.uint8)
            if rgb_window.ndim != 4 or rgb_window.shape[-1] != 3:
                raise ValueError(f"expected rgb_window [T,H,W,3], got {rgb_window.shape}")
            if rgb_window.shape[0] != expected_window_frames:
                raise ValueError(
                    f"checkpoint expects {expected_window_frames} online frames, got {rgb_window.shape[0]}"
                )
            # forward_policy always discards the other three latent planes and
            # replaces them by zeros.  Therefore only the oldest/reference RGB
            # can affect the output; hashing and encoding anything else would
            # change latency, not model semantics.
            reference_rgb = rgb_window[0]
            cache_key = hashlib.blake2b(reference_rgb.tobytes(), digest_size=16).digest()
            cached = inference_cache.get(cache_key)
            if cached is not None:
                connection.send(
                    {
                        **cached,
                        "cache_hit": True,
                        "preprocess_seconds": 0.0,
                        "vae_seconds": 0.0,
                        "policy_seconds": 0.0,
                        "total_seconds": time.perf_counter() - start,
                    }
                )
                continue
            frame = preprocess_rgb(reference_rgb, args.height, args.width)
            preprocess_seconds = time.perf_counter() - start

            # Existing GigaNav checkpoints were trained with labels starting
            # 12 steps after a T4 window's first frame.  A 13-frame rolling
            # observation window ending at the current environment state makes
            # that old first causal plane exactly t-12 and its first action
            # output exactly the action to execute at t.
            video = frame[:, None].float().div_(127.5).sub_(1.0)
            synchronize(device)
            vae_start = time.perf_counter()
            with torch.inference_mode():
                obs_latent = vae.encode(video[None].to(device=device, dtype=dtype))
                if obs_latent.shape[2] == 1:
                    obs_latent = torch.cat(
                        [
                            obs_latent,
                            obs_latent.new_zeros(
                                obs_latent.shape[0], obs_latent.shape[1], 3,
                                obs_latent.shape[3], obs_latent.shape[4],
                            ),
                        ],
                        dim=2,
                    )
                if obs_latent.shape[2] != 4:
                    raise RuntimeError(f"online Wan VAE returned {tuple(obs_latent.shape)}")
            synchronize(device)
            vae_seconds = time.perf_counter() - vae_start

            policy_start = time.perf_counter()
            with torch.inference_mode():
                output = model.forward_policy(
                    obs_latent=obs_latent,
                    text_embedding=text_embedding,
                    text_mask=text_mask,
                )
                probabilities = output["action_logits"].softmax(-1)
                sequence = probabilities.argmax(-1)[0]
            synchronize(device)
            policy_seconds = time.perf_counter() - policy_start
            response = {
                "status": "ok",
                "episode_id": current_episode,
                "action": int(sequence[0].item()),
                "action_sequence": sequence.cpu().tolist(),
                "first_action_probabilities": probabilities[0, 0].float().cpu().tolist(),
                "preprocess_seconds": preprocess_seconds,
                "vae_seconds": vae_seconds,
                "policy_seconds": policy_seconds,
                "total_seconds": time.perf_counter() - start,
                "latent_shape": list(obs_latent.shape),
                "training_reference_delay_steps": reference_delay,
                "online_vae_input_frames": expected_window_frames,
                "online_vae_encoded_frames": 1,
                "cache_hit": False,
            }
            inference_cache[cache_key] = response
            connection.send(response)
    except Exception as error:
        try:
            connection.send({"status": "error", "error": repr(error)})
        except Exception:
            pass
        raise
    finally:
        connection.close()
        listener.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
