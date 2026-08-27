"""Canonical Stage2 mixed-video data construction for the V1 pipeline.

The dataset is intentionally window-based rather than sample-list based:

``C_hist`` updates Register, ``Z_obs`` is the clean current main-stream token,
and ``Z_future_target`` is the video denoising target.  Pose supervision is
attached to the current chunk when a valid pose source is available.  Samples
without pose still participate in visual cotrain with ``pose_mask=0``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


FINAL_LATENT_T = 4
FINAL_LATENT_H = 56
FINAL_LATENT_W = 112
FINAL_MICRO_FRAMES = 13
FINAL_MICRO_STRIDE = 12
ACTION_BASE = 12
COMBO_DIM = ACTION_BASE * ACTION_BASE

__all__ = [
    "ACTION_BASE",
    "COMBO_DIM",
    "FINAL_LATENT_H",
    "FINAL_LATENT_T",
    "FINAL_LATENT_W",
    "FINAL_MICRO_FRAMES",
    "FINAL_MICRO_STRIDE",
    "FinalStage2BatchBuilder",
    "FinalStage2DataConfig",
    "Stage2TextCache",
    "TensorPayloadCache",
    "iter_jsonl",
    "latent_num_micro",
    "latent_path",
    "micro_steps_for_iw_chunks",
    "nearest_rotation",
    "parse_csv_ints",
    "parse_re10k_camera_file",
    "parse_weights",
    "re10k_scene_id",
    "relative_pose9",
    "rotmat_to_quat_wxyz",
]


@dataclass(slots=True)
class FinalStage2DataConfig:
    manifest: Path = Path("/sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes_spatial20.jsonl")
    latent_root: Path = Path("/sharedata/NAV/derived/v1/t4_micro_latents_spatial20")
    text_empty: Path = Path("/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt")
    text_cache_root: Path = Path("/sharedata/NAV/derived/v1/text_embeddings/t4_micro")
    re10k_camera_root: Path = Path("/sharedata/RealEstate10K/src_annotations/train/cameras")
    dataset_weights: str = "re10k=0.20,spatialvid=0.45,dl3dv=0.35"
    history_iw_chunks: str = "1,4,8,16"
    action_horizon: int = 10
    batch_size: int = 1
    empty_hist_prob: float = 0.10
    empty_cur_prob: float = 0.10
    payload_cache_items: int = 2
    text_cache_items: int = 128
    seed: int = 20260818

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("manifest", "latent_root", "text_empty", "text_cache_root", "re10k_camera_root"):
            payload[key] = str(payload[key])
        return payload


def parse_csv_ints(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"expected positive comma-separated integers, got {text!r}")
    return values


def parse_weights(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, value = part.split("=", 1)
        out[name.strip()] = float(value)
    if not out or sum(out.values()) <= 0:
        raise ValueError(f"invalid dataset weights: {text!r}")
    return out


def micro_steps_for_iw_chunks(iw_chunks: int) -> int:
    target_frames = 1 + 80 * int(iw_chunks)
    if target_frames <= FINAL_MICRO_FRAMES:
        return 1
    return math.ceil((target_frames - FINAL_MICRO_FRAMES) / FINAL_MICRO_STRIDE) + 1


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def latent_path(root: Path, dataset: str, sample_id: str) -> Path:
    return root / dataset / f"{sample_id}.pt"


def latent_num_micro(path: Path) -> int:
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        return int(json.loads(sidecar.read_text())["cached_micro_chunks"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("num_micro_chunks", payload["micro_latents"].shape[1]))


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation.astype(np.float32)


def rotmat_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    r = rotation.astype(np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(r)))
        if axis == 0:
            s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12)) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12)) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12)) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float32)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    if quat[0] < 0:
        quat = -quat
    return quat


def parse_re10k_camera_file(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    records: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return records
    with path.open() as stream:
        for line_no, line in enumerate(stream):
            if line_no == 0 and not line[:1].isdigit():
                continue
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            timestamp = int(float(parts[0]))
            intr = np.asarray([float(x) for x in parts[1:5]], dtype=np.float32)
            pose = np.asarray([float(x) for x in parts[5:17]], dtype=np.float32).reshape(3, 4)
            pose[:, :3] = nearest_rotation(pose[:, :3])
            records[timestamp] = (intr, pose)
    return records


def re10k_scene_id(sample_id: str) -> str:
    if sample_id.startswith("re10k__"):
        sample_id = sample_id[len("re10k__") :]
    return sample_id.split("_")[0]


def relative_pose9(
    records: dict[int, tuple[np.ndarray, np.ndarray]],
    timestamps: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        return np.zeros((len(timestamps), 9), dtype=np.float32), np.zeros((len(timestamps),), dtype=np.float32)
    keys = np.asarray(sorted(records), dtype=np.int64)
    poses: list[np.ndarray] = []
    fovs: list[np.ndarray] = []
    for timestamp in timestamps:
        if timestamp in records:
            intr, pose = records[timestamp]
        else:
            nearest = int(keys[np.argmin(np.abs(keys - int(timestamp)))])
            intr, pose = records[nearest]
        fx, fy = float(intr[0]), float(intr[1])
        fov_x = 2.0 * math.atan(0.5 / max(fx, 1e-8))
        fov_y = 2.0 * math.atan(0.5 / max(fy, 1e-8))
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :4] = pose
        poses.append(mat)
        fovs.append(np.asarray([fov_x, fov_y], dtype=np.float32))
    ref_inv = np.linalg.inv(poses[0])
    targets = []
    for pose, fov in zip(poses, fovs):
        rel = pose @ ref_inv
        targets.append(
            np.concatenate(
                [rel[:3, 3].astype(np.float32), rotmat_to_quat_wxyz(nearest_rotation(rel[:3, :3])), fov],
                axis=0,
            )
        )
    return np.stack(targets).astype(np.float32), np.ones((len(timestamps),), dtype=np.float32)


class TensorPayloadCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max(0, int(max_items))
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def load(self, path: Path) -> dict[str, Any]:
        key = str(path)
        if self.max_items and key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if self.max_items:
            self.cache[key] = payload
            while len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        return payload


class Stage2TextCache:
    def __init__(self, cfg: FinalStage2DataConfig) -> None:
        empty = torch.load(cfg.text_empty, map_location="cpu", weights_only=False)
        self.empty_y = empty["y"].to(torch.bfloat16)
        self.empty_mask = empty["y_mask"]
        self.root = cfg.text_cache_root
        self.max_items = max(0, int(cfg.text_cache_items))
        self.cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, str]] = OrderedDict()
        self.requests = 0
        self.fallbacks = 0
        self.hits = 0

    def load(self, dataset: str, sample_id: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        self.requests += 1
        key = f"{dataset}/{sample_id}"
        if self.max_items and key in self.cache:
            self.hits += 1
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        path = self.root / dataset / f"{sample_id}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            value = (payload["y"].to(torch.bfloat16), payload["y_mask"], str(path))
        else:
            self.fallbacks += 1
            value = (self.empty_y, self.empty_mask, "empty_text")
        if self.max_items:
            self.cache[key] = value
            while len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        return value

    def stats(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "hits": self.hits,
            "fallbacks": self.fallbacks,
            "cache_items": len(self.cache),
        }


class FinalStage2BatchBuilder:
    def __init__(self, cfg: FinalStage2DataConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.weights = parse_weights(cfg.dataset_weights)
        self.history_iw = parse_csv_ints(cfg.history_iw_chunks)
        self.history_micro_by_iw = {iw: micro_steps_for_iw_chunks(iw) for iw in self.history_iw}
        self.rows_by_sample: dict[str, dict[str, Any]] = {}
        self.windows: dict[int, dict[str, list[dict[str, Any]]]] = {iw: {} for iw in self.history_iw}
        self.cursors: dict[int, dict[str, int]] = {iw: {} for iw in self.history_iw}
        self.payload_cache = TensorPayloadCache(cfg.payload_cache_items)
        self.text_cache = Stage2TextCache(cfg)
        self.camera_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        self._build()

    def _build(self) -> None:
        for row in iter_jsonl(self.cfg.manifest):
            dataset = row.get("dataset")
            if dataset not in self.weights:
                continue
            sample_id = row["sample_id"]
            path = latent_path(self.cfg.latent_root, dataset, sample_id)
            if not path.is_file():
                continue
            num_micro = latent_num_micro(path)
            self.rows_by_sample[sample_id] = row
            for iw, history_micro in self.history_micro_by_iw.items():
                max_start = num_micro - history_micro - 2
                if max_start < 0:
                    continue
                bucket = self.windows[iw].setdefault(dataset, [])
                for start_micro in range(max_start + 1):
                    bucket.append(
                        {
                            "path": path,
                            "dataset": dataset,
                            "sample_id": sample_id,
                            "history_iw": iw,
                            "history_micro": history_micro,
                            "start_micro": start_micro,
                        }
                    )
        for iw, by_dataset in self.windows.items():
            for dataset, samples in by_dataset.items():
                self.rng.shuffle(samples)
                self.cursors[iw][dataset] = 0
        missing = [iw for iw, by_dataset in self.windows.items() if not any(by_dataset.values())]
        if missing:
            raise RuntimeError(f"no windows for IW histories: {missing}")

    def summary(self) -> dict[str, Any]:
        by_iw_dataset = {
            iw: {dataset: len(samples) for dataset, samples in by_dataset.items()}
            for iw, by_dataset in self.windows.items()
        }
        return {
            "config": self.cfg.to_dict(),
            "history_micro_by_iw": self.history_micro_by_iw,
            "num_rows": len(self.rows_by_sample),
            "num_windows_by_iw_dataset": by_iw_dataset,
            "num_windows_by_iw": {iw: sum(v.values()) for iw, v in by_iw_dataset.items()},
        }

    def _choose_history_iw(self) -> int:
        return self.rng.choice(self.history_iw)

    def _choose_window(self, iw: int) -> dict[str, Any]:
        by_dataset = self.windows[iw]
        available = [
            (dataset, self.weights.get(dataset, 0.0))
            for dataset, samples in by_dataset.items()
            if samples and self.weights.get(dataset, 0.0) > 0
        ]
        if not available:
            available = [(dataset, 1.0) for dataset, samples in by_dataset.items() if samples]
        dataset = self.rng.choices([x[0] for x in available], weights=[x[1] for x in available], k=1)[0]
        samples = by_dataset[dataset]
        cursor = self.cursors[iw][dataset]
        if cursor >= len(samples):
            self.rng.shuffle(samples)
            cursor = 0
        item = samples[cursor]
        self.cursors[iw][dataset] = cursor + 1
        return item

    def _camera(self, scene_id: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        if scene_id not in self.camera_cache:
            self.camera_cache[scene_id] = parse_re10k_camera_file(self.cfg.re10k_camera_root / f"{scene_id}.txt")
        return self.camera_cache[scene_id]

    def _action_window(self, values: torch.Tensor, frame_start: int) -> torch.Tensor:
        start = max(0, min(int(frame_start), max(int(values.numel()) - 1, 0)))
        end = min(start + self.cfg.action_horizon, int(values.numel()))
        out = values[start:end].long()
        if out.numel() < self.cfg.action_horizon:
            pad = out[-1:] if out.numel() else torch.zeros(1, dtype=torch.long)
            out = torch.cat([out, pad.expand(self.cfg.action_horizon - out.numel())], dim=0)
        return out[: self.cfg.action_horizon].clamp_(0, ACTION_BASE - 1)

    def _pose_for_window(self, item: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = item["dataset"]
        if dataset != "re10k":
            return torch.zeros(FINAL_LATENT_T, 9, dtype=torch.float32), torch.zeros(FINAL_LATENT_T, dtype=torch.float32)
        row = self.rows_by_sample[item["sample_id"]]
        frame_paths = row.get("frame_paths") or []
        if not frame_paths:
            return torch.zeros(FINAL_LATENT_T, 9, dtype=torch.float32), torch.zeros(FINAL_LATENT_T, dtype=torch.float32)
        current_frame_start = (int(item["start_micro"]) + int(item["history_micro"])) * FINAL_MICRO_STRIDE
        anchors = np.rint(np.linspace(0, FINAL_MICRO_FRAMES - 1, FINAL_LATENT_T)).astype(int).tolist()
        timestamps: list[int] = []
        for anchor in anchors:
            idx = max(0, min(current_frame_start + int(anchor), len(frame_paths) - 1))
            timestamps.append(int(Path(frame_paths[idx]).stem))
        pose, mask = relative_pose9(self._camera(re10k_scene_id(item["sample_id"])), timestamps)
        return torch.from_numpy(pose), torch.from_numpy(mask)

    def next_batch(self, *, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        items = [self._choose_window(self._choose_history_iw()) for _ in range(self.cfg.batch_size)]
        history_micro = int(items[0]["history_micro"])
        # Keep tensor shapes uniform inside a micro-batch.
        if any(int(x["history_micro"]) != history_micro for x in items):
            iw = int(items[0]["history_iw"])
            items = [self._choose_window(iw) for _ in range(self.cfg.batch_size)]
        histories = []
        z_obs = []
        z_target = []
        a_hist = []
        a_cur = []
        poses = []
        pose_masks = []
        ys = []
        y_masks = []
        empty_hist_flags = []
        empty_cur_flags = []
        datasets = []
        for item in items:
            payload = self.payload_cache.load(item["path"])
            chunks = payload["micro_latents"][0]
            if tuple(chunks.shape[1:]) != (16, FINAL_LATENT_T, FINAL_LATENT_H, FINAL_LATENT_W):
                raise RuntimeError(f"bad latent shape for {item['sample_id']}: {tuple(chunks.shape)}")
            start = int(item["start_micro"])
            histories.append(chunks[start : start + history_micro].float())
            z_obs.append(chunks[start + history_micro].float())
            z_target.append(chunks[start + history_micro + 1].float())
            move = payload["move"].long()
            view = payload.get("view", move).long()
            hist_combo = []
            for offset in range(history_micro):
                frame_start = (start + offset) * FINAL_MICRO_STRIDE
                trans = self._action_window(move, frame_start)
                rot = self._action_window(view, frame_start)
                hist_combo.append((trans * ACTION_BASE + rot).clamp_(0, COMBO_DIM - 1))
            hist_combo_t = torch.stack(hist_combo, dim=0)
            cur_frame_start = (start + history_micro) * FINAL_MICRO_STRIDE
            cur_trans = self._action_window(move, cur_frame_start)
            cur_rot = self._action_window(view, cur_frame_start)
            cur_combo = (cur_trans * ACTION_BASE + cur_rot).clamp_(0, COMBO_DIM - 1)
            empty_hist = self.rng.random() < self.cfg.empty_hist_prob
            empty_cur = self.rng.random() < self.cfg.empty_cur_prob
            if empty_hist:
                hist_combo_t.zero_()
            if empty_cur:
                cur_combo.zero_()
            a_hist.append(hist_combo_t)
            a_cur.append(cur_combo)
            pose, pose_mask = self._pose_for_window(item)
            poses.append(pose)
            pose_masks.append(pose_mask)
            y, y_mask, _ = self.text_cache.load(item["dataset"], item["sample_id"])
            ys.append(y)
            y_masks.append(y_mask)
            empty_hist_flags.append(empty_hist)
            empty_cur_flags.append(empty_cur)
            datasets.append(item["dataset"])
        history_latents = torch.stack(histories, dim=0)
        z_obs_t = torch.stack(z_obs, dim=0)
        z_target_t = torch.stack(z_target, dim=0)
        z_noise = torch.randn_like(z_target_t)
        sigma = torch.rand(self.cfg.batch_size)
        visual_t = 7.0 * sigma / (1.0 + 6.0 * sigma)
        z_noisy = (1.0 - visual_t.view(-1, 1, 1, 1, 1)) * z_target_t + visual_t.view(-1, 1, 1, 1, 1) * z_noise
        batch = {
            "history_latents": history_latents.to(device=device, dtype=dtype, non_blocking=True),
            "z_obs": z_obs_t.to(device=device, dtype=dtype, non_blocking=True),
            "z_future_target": z_target_t.to(device=device, dtype=dtype, non_blocking=True),
            "z_future_noise": z_noise.to(device=device, dtype=dtype, non_blocking=True),
            "z_future_noisy": z_noisy.to(device=device, dtype=dtype, non_blocking=True),
            "visual_timestep": visual_t.to(device=device, dtype=torch.float32, non_blocking=True),
            "a_hist_combo": torch.stack(a_hist, dim=0).to(device=device, non_blocking=True),
            "a_cur_combo": torch.stack(a_cur, dim=0).to(device=device, non_blocking=True),
            "a_noise": torch.randn(self.cfg.batch_size, self.cfg.action_horizon, 6, device=device, dtype=dtype),
            "action_timestep": torch.rand(self.cfg.batch_size, device=device, dtype=torch.float32),
            "y": torch.cat(ys, dim=0).to(device=device, dtype=dtype, non_blocking=True),
            "y_mask": torch.cat(y_masks, dim=0).to(device=device, dtype=dtype, non_blocking=True),
            "pose_target": torch.stack(poses, dim=0).to(device=device, dtype=torch.float32, non_blocking=True),
            "pose_mask": torch.stack(pose_masks, dim=0).to(device=device, dtype=torch.float32, non_blocking=True),
        }
        meta = {
            "datasets": datasets,
            "history_iw": int(items[0]["history_iw"]),
            "history_micro": history_micro,
            "pose_valid_frames": float(batch["pose_mask"].detach().sum().cpu().item()),
            "empty_hist_count": int(sum(empty_hist_flags)),
            "empty_cur_count": int(sum(empty_cur_flags)),
            "text_cache": self.text_cache.stats(),
        }
        return batch, meta
