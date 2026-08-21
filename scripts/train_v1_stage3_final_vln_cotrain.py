#!/usr/bin/env python3
"""Final Stage3 VLN cotrain from a final Stage2 checkpoint.

The policy branch uses prepared VLN T4 micro-latents:

``history_latents + A_hist -> Register; Z_obs + text + A_noise -> shared Wan action tokens``

and supervises the policy output with unified trans/rot combo labels.  Stage2
video/pose replay is mixed in each optimizer step to reduce representation
forgetting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import gzip
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nav.v1.stage2_final import FinalStage2BatchBuilder, FinalStage2DataConfig, FinalStage2WanConfig, FinalStage2WanModel  # noqa: E402
from nav.v1.stage2_final.data import ACTION_BASE, COMBO_DIM, FINAL_LATENT_H, FINAL_LATENT_T, FINAL_LATENT_W, micro_steps_for_iw_chunks  # noqa: E402
from train_v1_stage2_final_cotrain import install_non_reentrant_wan_checkpoint, now, torch_dtype, write_json  # noqa: E402


DEFAULT_STAGE2_CHECKPOINT = Path(
    "/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/log/v1_stage2_final_cotrain/"
    "stage2_final_branchmask_policyreg_venv_iw14816_20260819_015948/checkpoints/step_002000.pt"
)
DEFAULT_VLN_MANIFEST = Path(
    "/sharedata/NAV/derived/v1/vln/t4_micro_latents_500g/rxr_budget500_stream_20260821_0000/"
    "manifests/encoded_episodes.jsonl"
)


@dataclass(slots=True)
class TrainConfig:
    run_name: str
    checkpoint: str
    vln_manifest: str
    device: str
    steps: int
    batch_size: int
    grad_accum: int
    effective_batch_size: int
    history_iw_chunks: str
    lr: float
    weight_decay: float
    grad_clip: float
    lambda_ce: float
    lambda_video_replay: float
    lambda_pose_replay: float
    dtype: str
    save_every: int
    log_every: int
    seed: int
    output_root: str
    vln_text_cache_root: str
    require_vln_text_cache: bool


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def make_dataclass(cls, payload: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    clean = {k: v for k, v in payload.items() if k in allowed}
    for key in ("manifest", "latent_root", "text_empty", "text_cache_root", "re10k_camera_root"):
        if key in clean:
            clean[key] = Path(clean[key])
    if cls is FinalStage2WanConfig and "register_condition_grid" in clean:
        clean["register_condition_grid"] = tuple(clean["register_condition_grid"])
    return cls(**clean)


def load_stage2_model(checkpoint: Path, *, device: torch.device, dtype: torch.dtype) -> tuple[FinalStage2WanModel, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = make_dataclass(FinalStage2WanConfig, ckpt["model_config"])
    model = FinalStage2WanModel(cfg)
    model.remove_hmpc()
    result = model.load_state_dict(ckpt["model"], strict=False)
    bad_missing = [key for key in result.missing_keys if not key.startswith("backbone.latent_encoder.")]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing[:20]} unexpected={result.unexpected_keys[:20]}")
    return model.to(device=device, dtype=dtype).train(), ckpt


def trans_rot_to_combo(trans_id: int, rot_id: int) -> int:
    trans_id = max(0, min(int(trans_id), ACTION_BASE - 1))
    rot_id = max(0, min(int(rot_id), ACTION_BASE - 1))
    return trans_id * ACTION_BASE + rot_id


def vln_action_to_combo(action_id: int) -> int:
    return {
        0: trans_rot_to_combo(10, 0),  # STOP
        1: trans_rot_to_combo(1, 0),   # MOVE_FORWARD
        2: trans_rot_to_combo(0, 3),   # TURN_LEFT
        3: trans_rot_to_combo(0, 4),   # TURN_RIGHT
    }.get(int(action_id), 0)


class VLNT4PolicyDataset:
    def __init__(
        self,
        *,
        manifest: Path,
        history_iw_chunks: str,
        action_horizon: int,
        batch_size: int,
        text_empty: Path,
        text_cache_root: Path,
        require_text_cache: bool,
        max_episodes: int,
        seed: int,
    ) -> None:
        self.rng = random.Random(seed)
        self.history_iw = [int(x.strip()) for x in history_iw_chunks.split(",") if x.strip()]
        self.history_micro_by_iw = {iw: micro_steps_for_iw_chunks(iw) for iw in self.history_iw}
        self.action_horizon = int(action_horizon)
        self.batch_size = int(batch_size)
        empty = torch.load(text_empty, map_location="cpu", weights_only=False)
        self.empty_y = empty["y"].to(torch.bfloat16)
        self.empty_mask = empty["y_mask"]
        self.text_cache_root = text_cache_root
        self.require_text_cache = bool(require_text_cache)
        self.text_cache: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
        self.text_hits = 0
        self.text_fallbacks = 0
        rows = []
        with open_text(manifest) as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") not in {None, "encoded"}:
                    continue
                path = Path(row.get("latent_path", ""))
                if path.is_file():
                    rows.append({"latent_path": path, "sample_id": row.get("sample_id", path.stem)})
        if max_episodes > 0:
            rows = rows[:max_episodes]
        self.rows = rows
        self.windows: dict[int, list[dict[str, Any]]] = {iw: [] for iw in self.history_iw}
        for row in rows:
            sidecar = row["latent_path"].with_suffix(".json")
            if sidecar.is_file():
                try:
                    meta = json.loads(sidecar.read_text())
                    num_micro = int(
                        meta.get("num_micro_chunks")
                        or meta.get("cached_micro_chunks")
                        or meta.get("source_num_micro_chunks")
                        or 0
                    )
                except Exception:
                    num_micro = 0
            else:
                num_micro = 0
            if num_micro <= 0:
                try:
                    payload = torch.load(row["latent_path"], map_location="cpu", weights_only=False)
                    num_micro = int(payload.get("num_micro_chunks", payload["micro_latents"].shape[1]))
                except Exception:
                    num_micro = 0
            for iw, history_micro in self.history_micro_by_iw.items():
                max_start = int(num_micro) - history_micro - 1
                for start in range(max(0, max_start + 1)):
                    self.windows[iw].append({**row, "history_iw": iw, "history_micro": history_micro, "start_micro": start})
        for samples in self.windows.values():
            self.rng.shuffle(samples)
        self.cursors = {iw: 0 for iw in self.history_iw}
        if not any(self.windows.values()):
            raise RuntimeError(f"no usable VLN T4 windows in {manifest}; histories={self.history_micro_by_iw}")
        self.cache: dict[str, dict[str, Any]] = {}

    def summary(self) -> dict[str, Any]:
        return {
            "episodes": len(self.rows),
            "history_micro_by_iw": self.history_micro_by_iw,
            "windows_by_iw": {iw: len(samples) for iw, samples in self.windows.items()},
        }

    def _payload(self, path: Path) -> dict[str, Any]:
        key = str(path)
        if key not in self.cache:
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
            if len(self.cache) > 2:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def _choose_iw(self) -> int:
        available = [iw for iw, samples in self.windows.items() if samples]
        return self.rng.choice(available)

    def _choose_window(self, iw: int) -> dict[str, Any]:
        samples = self.windows[iw]
        cursor = self.cursors[iw]
        if cursor >= len(samples):
            self.rng.shuffle(samples)
            cursor = 0
        self.cursors[iw] = cursor + 1
        return samples[cursor]

    def _actions(self, payload: dict[str, Any]) -> list[int]:
        action_path = payload.get("action_path")
        if action_path and Path(action_path).is_file():
            data = json.loads(Path(action_path).read_text())
            return [int(x) for x in data.get("gt_actions", [])]
        return []

    def _text(self, payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, str]:
        dataset = str(payload.get("dataset", "unknown"))
        sample_id = str(payload.get("sample_id", ""))
        key = f"{dataset}/{sample_id}"
        if key in self.text_cache:
            y, y_mask, source = self.text_cache[key]
            self.text_hits += 1
            return y, y_mask, source
        path = self.text_cache_root / dataset / f"{sample_id}.pt"
        if path.is_file():
            text_payload = torch.load(path, map_location="cpu", weights_only=False)
            y = text_payload["y"].to(torch.bfloat16)
            y_mask = text_payload["y_mask"]
            source = str(path)
            self.text_hits += 1
        else:
            if self.require_text_cache:
                raise FileNotFoundError(f"missing VLN instruction embedding: {path}")
            y = self.empty_y
            y_mask = self.empty_mask
            source = "empty_fallback_missing_vln_instruction_embedding"
            self.text_fallbacks += 1
        self.text_cache[key] = (y, y_mask, source)
        if len(self.text_cache) > 128:
            self.text_cache.pop(next(iter(self.text_cache)))
        return y, y_mask, source

    def _action_window(self, actions: list[int], start: int) -> tuple[torch.Tensor, torch.Tensor]:
        combos = []
        mask = []
        for index in range(start, start + self.action_horizon):
            if 0 <= index < len(actions):
                combos.append(vln_action_to_combo(actions[index]))
                mask.append(1.0)
            else:
                combos.append(0)
                mask.append(0.0)
        return torch.tensor(combos, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

    def next_batch(self, *, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        iw = self._choose_iw()
        items = [self._choose_window(iw) for _ in range(self.batch_size)]
        histories = []
        z_obs = []
        a_hist = []
        action_combo = []
        action_masks = []
        ys = []
        y_masks = []
        text_sources = []
        sample_ids = []
        for item in items:
            payload = self._payload(item["latent_path"])
            chunks = payload["micro_latents"][0].float()
            if tuple(chunks.shape[1:]) != (16, FINAL_LATENT_T, FINAL_LATENT_H, FINAL_LATENT_W):
                raise RuntimeError(f"bad latent shape for {item['latent_path']}: {tuple(chunks.shape)}")
            start = int(item["start_micro"])
            history_micro = int(item["history_micro"])
            histories.append(chunks[start : start + history_micro])
            z_obs.append(chunks[start + history_micro])
            actions = self._actions(payload)
            hist_rows = []
            for offset in range(history_micro):
                combo, _mask = self._action_window(actions, (start + offset) * 12)
                hist_rows.append(combo)
            a_hist.append(torch.stack(hist_rows, dim=0))
            combo, mask = self._action_window(actions, (start + history_micro) * 12)
            action_combo.append(combo)
            action_masks.append(mask)
            y, y_mask, text_source = self._text(payload)
            ys.append(y)
            y_masks.append(y_mask)
            text_sources.append(text_source)
            sample_ids.append(str(payload.get("sample_id", item["sample_id"])))
        z_obs_t = torch.stack(z_obs, dim=0)
        y = torch.cat(ys, dim=0)
        y_mask = torch.cat(y_masks, dim=0)
        batch = {
            "history_latents": torch.stack(histories, dim=0).to(device=device, dtype=dtype, non_blocking=True),
            "z_obs": z_obs_t.to(device=device, dtype=dtype, non_blocking=True),
            "z_future_noisy": torch.zeros_like(z_obs_t).to(device=device, dtype=dtype, non_blocking=True),
            "visual_timestep": torch.zeros(self.batch_size, device=device, dtype=torch.float32),
            "a_hist_combo": torch.stack(a_hist, dim=0).to(device=device, non_blocking=True),
            "a_cur_combo": torch.zeros(self.batch_size, self.action_horizon, device=device, dtype=torch.long),
            "a_noise": torch.randn(self.batch_size, self.action_horizon, 6, device=device, dtype=dtype),
            "action_timestep": torch.rand(self.batch_size, device=device, dtype=torch.float32),
            "action_combo": torch.stack(action_combo, dim=0).to(device=device, non_blocking=True).clamp_(0, COMBO_DIM - 1),
            "action_loss_mask": torch.stack(action_masks, dim=0).to(device=device, dtype=torch.float32, non_blocking=True),
            "action_target": torch.zeros(self.batch_size, self.action_horizon, 6, device=device, dtype=dtype),
            "y": y.to(device=device, dtype=dtype, non_blocking=True),
            "y_mask": y_mask.to(device=device, dtype=dtype, non_blocking=True),
        }
        meta = {
            "history_iw": iw,
            "history_micro": self.history_micro_by_iw[iw],
            "sample_ids": sample_ids,
            "action_valid": float(batch["action_loss_mask"].sum().detach().cpu().item()),
            "text_sources": text_sources,
            "text_hits": self.text_hits,
            "text_fallbacks": self.text_fallbacks,
        }
        return batch, meta


def save_checkpoint(
    *,
    model: FinalStage2WanModel,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    step: int,
    train_cfg: TrainConfig,
    model_cfg: FinalStage2WanConfig,
    data_cfg: FinalStage2DataConfig,
    source_checkpoint: str,
) -> Path:
    path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_config": asdict(train_cfg),
            "model_config": model_cfg.to_dict(),
            "data_config": data_cfg.to_dict(),
            "source_checkpoint": source_checkpoint,
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"stage3_final_vln_cotrain_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE2_CHECKPOINT)
    parser.add_argument("--vln-manifest", type=Path, default=DEFAULT_VLN_MANIFEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--history-iw-chunks", default="1")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-video-replay", type=float, default=0.25)
    parser.add_argument("--lambda-pose-replay", type=float, default=0.05)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-vln-episodes", type=int, default=0)
    parser.add_argument("--vln-text-cache-root", type=Path, default=Path("/sharedata/NAV/derived/v1/vln/text_embeddings/t4_micro"))
    parser.add_argument("--require-vln-text-cache", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "log" / "v1_stage3_final_vln_cotrain")
    args = parser.parse_args()

    if args.batch_size * args.grad_accum != 16:
        raise SystemExit("final Stage3 cotrain requires effective batch size = 16")
    os.environ.setdefault("NAV_INF_WORLD_ROOT", str(ROOT.parent / "Infinite-World"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_non_reentrant_wan_checkpoint()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    model, source_ckpt = load_stage2_model(args.checkpoint, device=device, dtype=dtype)
    data_cfg = make_dataclass(FinalStage2DataConfig, source_ckpt["data_config"])
    data_cfg.batch_size = args.batch_size
    data_cfg.seed = args.seed + 1000
    replay_builder = FinalStage2BatchBuilder(data_cfg)
    vln_builder = VLNT4PolicyDataset(
        manifest=args.vln_manifest,
        history_iw_chunks=args.history_iw_chunks,
        action_horizon=model.cfg.action_horizon,
        batch_size=args.batch_size,
        text_empty=data_cfg.text_empty,
        text_cache_root=args.vln_text_cache_root,
        require_text_cache=args.require_vln_text_cache,
        max_episodes=args.max_vln_episodes,
        seed=args.seed,
    )
    train_cfg = TrainConfig(
        run_name=args.run_name,
        checkpoint=str(args.checkpoint),
        vln_manifest=str(args.vln_manifest),
        device=str(device),
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        effective_batch_size=args.batch_size * args.grad_accum,
        history_iw_chunks=args.history_iw_chunks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_ce=args.lambda_ce,
        lambda_video_replay=args.lambda_video_replay,
        lambda_pose_replay=args.lambda_pose_replay,
        dtype=args.dtype,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        output_root=str(args.output_root),
        vln_text_cache_root=str(args.vln_text_cache_root),
        require_vln_text_cache=bool(args.require_vln_text_cache),
    )
    preflight = {
        "event": "stage3_final_vln_cotrain_preflight",
        "time": now(),
        "train": asdict(train_cfg),
        "vln": vln_builder.summary(),
        "replay_data": replay_builder.summary(),
        "model": model.structural_report(),
        "source_checkpoint_step": int(source_ckpt["step"]),
        "policy_branch": "Register + Z_obs + empty/text + A_noise; no A_cur in policy forward",
        "loss": "L = lambda_ce * CE(combo_logits, action_combo) + lambda_video_replay * L_visual + lambda_pose_replay * L_pose",
        "text": {
            "vln_text_cache_root": str(args.vln_text_cache_root),
            "require_vln_text_cache": bool(args.require_vln_text_cache),
            "rule": "load per-sample VLN instruction UMT5 embedding; fallback to empty only when cache is missing",
        },
    }
    write_json(run_dir / "formal_preflight.json", preflight)
    write_json(run_dir / "config.json", {"train": asdict(train_cfg), "stage2_data": data_cfg.to_dict(), "model": model.cfg.to_dict()})
    try:
        from infworld.models.checkpoint import set_grad_checkpoint

        set_grad_checkpoint(model.backbone)
    except Exception:
        pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    log_path = run_dir / "train.jsonl"
    with log_path.open("a") as log:
        log.write(json.dumps({"event": "start", "time": now(), "run_dir": str(run_dir), **preflight}, ensure_ascii=False) + "\n")
        last = time.time()
        global_start = time.time()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            metrics: dict[str, float] = {}
            vln_meta_last: dict[str, Any] = {}
            replay_meta_last: dict[str, Any] = {}
            for _ in range(args.grad_accum):
                vln_batch, vln_meta = vln_builder.next_batch(device=device, dtype=dtype)
                out = model.forward_stage3_policy(vln_batch, lambda_ce=args.lambda_ce)
                loss = out["loss"] / args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite policy loss at step {step}: {loss}")
                loss.backward()
                metrics["loss_policy"] = metrics.get("loss_policy", 0.0) + float(out["loss"].detach().float().cpu().item()) / args.grad_accum
                metrics["loss_ce_aux"] = metrics.get("loss_ce_aux", 0.0) + float(out["loss_ce_aux"].detach().float().cpu().item()) / args.grad_accum
                metrics["loss_action_flow"] = metrics.get("loss_action_flow", 0.0) + float(out["loss_action_flow"].detach().float().cpu().item()) / args.grad_accum
                vln_meta_last = vln_meta

                replay_batch, replay_meta = replay_builder.next_batch(device=device, dtype=dtype)
                replay_out = model.forward_stage2(replay_batch, lambda_pose=args.lambda_pose_replay)
                replay_loss = (
                    float(args.lambda_video_replay) * replay_out["loss_visual_tensor"]
                    + float(args.lambda_pose_replay) * replay_out["loss_pose_tensor"]
                ) / args.grad_accum
                if not torch.isfinite(replay_loss):
                    raise FloatingPointError(f"non-finite replay loss at step {step}: {replay_loss}")
                replay_loss.backward()
                metrics["loss_video_replay"] = metrics.get("loss_video_replay", 0.0) + float(replay_out["loss_visual"].detach().float().cpu().item()) / args.grad_accum
                metrics["loss_pose_replay"] = metrics.get("loss_pose_replay", 0.0) + float(replay_out["loss_pose"].detach().float().cpu().item()) / args.grad_accum
                replay_meta_last = replay_meta
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step % args.log_every == 0 or step == 1:
                now_time = time.time()
                record = {
                    "event": "train",
                    "time": now(),
                    "step": step,
                    "seconds_per_step_window": (now_time - last) / max(args.log_every, 1),
                    "seconds_total": now_time - global_start,
                    "lr": optimizer.param_groups[0]["lr"],
                    "vln_history_iw": vln_meta_last.get("history_iw"),
                    "vln_action_valid": vln_meta_last.get("action_valid"),
                    "vln_text_hits": vln_meta_last.get("text_hits"),
                    "vln_text_fallbacks": vln_meta_last.get("text_fallbacks"),
                    "replay_history_iw": replay_meta_last.get("history_iw"),
                    "replay_datasets": replay_meta_last.get("datasets"),
                    **{f"train/{k}": v for k, v in metrics.items()},
                }
                if device.type == "cuda":
                    record["cuda_max_memory_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                last = now_time
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, step)
            if step % args.save_every == 0 or step == args.steps:
                ckpt = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    run_dir=run_dir,
                    step=step,
                    train_cfg=train_cfg,
                    model_cfg=model.cfg,
                    data_cfg=data_cfg,
                    source_checkpoint=str(args.checkpoint),
                )
                log.write(json.dumps({"event": "checkpoint", "time": now(), "step": step, "path": str(ckpt)}, ensure_ascii=False) + "\n")
                log.flush()
    writer.close()
    print(json.dumps({"status": "ok", "run_dir": str(run_dir), "log": str(log_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
