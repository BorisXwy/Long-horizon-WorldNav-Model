#!/usr/bin/env python3
"""Train V1 Stage One with T_latent=4 micro chunks and IW-aligned history.

History length is specified in InfiniteWorld-equivalent chunks. The script
converts it to T4 micro update steps:
  micro_frames=13, micro_stride=12
  IW N chunks = 1 + 80N RGB frames
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "Infinite-World"))

import infworld.context_parallel.context_parallel_util as cp_util
import infworld.models.dit_model as dit_model_module
from infworld.models.checkpoint import set_grad_checkpoint
from infworld.models.scheduler import RFlowScheduler, timestep_transform
from nav.infinite_adapter import InfiniteRegisterAdapter


DEFAULT_CKPT = Path("/sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
DEFAULT_TEXT = Path("/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt")
DEFAULT_DATA = Path("/sharedata/NAV/derived/v1/t4_micro_latents")
DEFAULT_TEXT_CACHE_ROOT = Path("/sharedata/NAV/derived/v1/text_embeddings/t4_micro")


def install_non_reentrant_checkpoint() -> None:
    def checkpoint_module(module, *args, **kwargs):
        if getattr(module, "grad_checkpointing", False):
            return torch.utils.checkpoint.checkpoint(
                module, *args, use_reentrant=False, **kwargs
            )
        return module(*args, **kwargs)

    dit_model_module.auto_grad_checkpoint = checkpoint_module


def micro_steps_for_iw_chunks(iw_chunks: int, *, micro_frames: int, micro_stride: int) -> int:
    target_frames = 1 + 80 * iw_chunks
    if target_frames <= micro_frames:
        return 1
    return math.ceil((target_frames - micro_frames) / micro_stride) + 1


def parse_iw_history(text: str) -> list[int]:
    values = [int(x) for x in text.split(",") if x.strip()]
    if not values or any(x < 1 for x in values):
        raise ValueError("--history-iw-chunks 必须包含正整数")
    return values


def parse_dataset_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--dataset-weights 项必须是 name=value: {part}")
        name, value = part.split("=", 1)
        name = name.strip()
        weight = float(value)
        if not name or weight < 0:
            raise ValueError(f"非法 dataset weight: {part}")
        weights[name] = weight
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("--dataset-weights 至少需要一个正权重")
    return weights


def parse_name_set(text: str) -> set[str]:
    return {x.strip() for x in text.split(",") if x.strip()}


def make_model(*, checkpoint: Path, device: torch.device, variant: str, train_scope: str) -> InfiniteRegisterAdapter:
    model = InfiniteRegisterAdapter(
        variant=variant,
        model_cfg={
            "model_type": "t2v",
            "dim": 1536,
            "in_channels": 16,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "num_heads": 12,
            "num_layers": 30,
            "out_channels": 16,
            "caption_channels": 4096,
            "model_max_length": 512,
        },
        register_cfg=(
            {
                "channels": 16,
                "register_frames": 4,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_update_layers": 2,
                "chunk_time_tokens": 4,
            }
            if variant == "latent_prefix"
            else {
                "latent_channels": 16,
                "register_dim": 256,
                "num_registers": 16,
                "num_update_layers": 2,
                "num_heads": 8,
                "caption_channels": 4096,
            }
        ),
    )
    audit = model.load_infinite_checkpoint(str(checkpoint))
    model._checkpoint_audit = audit
    if train_scope == "register":
        model.freeze_backbone()
    elif train_scope == "full":
        model.requires_grad_(True)
    else:
        raise ValueError(train_scope)
    model.to(device=device, dtype=torch.bfloat16).train()
    set_grad_checkpoint(model.backbone)
    return model


def latent_num_micro(path: Path) -> int:
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        return int(json.loads(sidecar.read_text())["cached_micro_chunks"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("num_micro_chunks", payload["micro_latents"].shape[1]))


def eligible_files(data_root: Path, required_micro: int) -> list[Path]:
    files = sorted(data_root.rglob("*.pt"))
    out = []
    for path in files:
        num_micro = latent_num_micro(path)
        if num_micro >= required_micro:
            out.append(path)
    return out


def build_window_buckets(
    data_root: Path,
    history_micro_by_iw: dict[int, int],
) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], dict[int, int], dict[int, dict[str, int]], list[Path]]:
    return build_window_buckets_filtered(
        data_root,
        history_micro_by_iw,
        text_cache_root=None,
        require_text_datasets=set(),
    )[:4]


def build_window_buckets_filtered(
    data_root: Path,
    history_micro_by_iw: dict[int, int],
    *,
    text_cache_root: Path | None,
    require_text_datasets: set[str],
) -> tuple[
    dict[int, dict[str, list[dict[str, Any]]]],
    dict[int, int],
    dict[int, dict[str, int]],
    list[Path],
    dict[str, Any],
]:
    files = sorted(data_root.rglob("*.pt"))
    buckets: dict[int, dict[str, list[dict[str, Any]]]] = {iw: {} for iw in history_micro_by_iw}
    used_files: set[Path] = set()
    skipped_text_required: dict[str, int] = {}
    for path in files:
        num_micro = latent_num_micro(path)
        dataset = path.parent.name
        sample_id = path.stem
        if dataset in require_text_datasets:
            if text_cache_root is None or not (text_cache_root / dataset / f"{sample_id}.pt").is_file():
                skipped_text_required[dataset] = skipped_text_required.get(dataset, 0) + 1
                continue
        for iw, history_micro in history_micro_by_iw.items():
            max_start = num_micro - history_micro - 1
            if max_start < 0:
                continue
            used_files.add(path)
            dataset_bucket = buckets[iw].setdefault(dataset, [])
            for start_micro in range(max_start + 1):
                dataset_bucket.append({
                    "path": path,
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "history_iw": iw,
                    "history_micro": history_micro,
                    "start_micro": start_micro,
                })
    counts_by_dataset = {
        iw: {dataset: len(samples) for dataset, samples in by_dataset.items()}
        for iw, by_dataset in buckets.items()
    }
    counts = {
        iw: sum(by_dataset.values())
        for iw, by_dataset in counts_by_dataset.items()
    }
    filter_stats = {
        "candidate_files": len(files),
        "used_files": len(used_files),
        "require_text_datasets": sorted(require_text_datasets),
        "skipped_files_missing_required_text": skipped_text_required,
    }
    return buckets, counts, counts_by_dataset, sorted(used_files), filter_stats


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def pad_action_81(values: torch.Tensor, start: int, length: int) -> torch.Tensor:
    out = torch.zeros(81, dtype=torch.long)
    if values.numel() == 0:
        return out
    segment = values[start : min(start + length, values.numel())].long()
    if segment.numel():
        out[-segment.numel():] = segment
    return out


def sample_visual_timestep(
    scheduler: RFlowScheduler,
    x_start: torch.Tensor,
) -> torch.Tensor:
    """Sample the visual diffusion/flow timestep exactly like RFlowScheduler.

    We sample it outside `training_losses` so the same step can pass a separate
    Giga-style `action_timestep` into A_query/action tokens.
    """

    if scheduler.use_discrete_timesteps:
        t = torch.randint(
            0,
            scheduler.num_timesteps,
            (x_start.shape[0],),
            device=x_start.device,
        )
    elif scheduler.sample_method == "uniform":
        t = torch.rand((x_start.shape[0],), device=x_start.device) * scheduler.num_timesteps
    elif scheduler.sample_method == "logit-normal":
        t = scheduler.sample_t(x_start) * scheduler.num_timesteps
    else:
        raise ValueError(f"unknown sample_method={scheduler.sample_method}")

    if scheduler.use_timestep_transform:
        t = timestep_transform(
            t,
            shift=scheduler.shift,
            num_timesteps=scheduler.num_timesteps,
        )
    return t


def sample_action_timestep(
    batch_size: int,
    *,
    device: torch.device,
    shift: float,
    num_timesteps: int = 1000,
) -> torch.Tensor:
    """Giga-style action-token timestep sampling.

    Action tokens have their own timestep, separate from the visual target
    timestep.  Clean prefix/history/local tokens stay at t=0.
    """

    sigma = torch.rand(batch_size, device=device)
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    timestep = torch.round(sigma * num_timesteps)
    return timestep.to(torch.float32)


class TextConditionCache:
    def __init__(
        self,
        *,
        empty_path: Path,
        text_cache_root: Path | None,
        device: torch.device,
        max_items: int,
    ) -> None:
        empty = torch.load(empty_path, map_location="cpu", weights_only=False)
        self.empty_y_cpu = empty["y"].to(torch.bfloat16)
        self.empty_y_mask_cpu = empty["y_mask"]
        self.text_cache_root = text_cache_root
        self.device = device
        self.max_items = max(0, max_items)
        self.cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, str]] = OrderedDict()
        self.requests = 0
        self.hits = 0
        self.fallbacks = 0

    def _load_one_cpu(self, dataset: str, sample_id: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        self.requests += 1
        key = f"{dataset}/{sample_id}"
        if key in self.cache:
            self.hits += 1
            y, y_mask, source = self.cache.pop(key)
            self.cache[key] = (y, y_mask, source)
            return y, y_mask, source
        path = None
        if self.text_cache_root is not None:
            candidate = self.text_cache_root / dataset / f"{sample_id}.pt"
            if candidate.is_file():
                path = candidate
        if path is None:
            self.fallbacks += 1
            y, y_mask, source = self.empty_y_cpu, self.empty_y_mask_cpu, "empty_fallback"
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            y = payload["y"].to(torch.bfloat16)
            y_mask = payload["y_mask"]
            source = str(path)
        if self.max_items:
            self.cache[key] = (y, y_mask, source)
            while len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        return y, y_mask, source

    def batch(self, items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        ys = []
        masks = []
        sources = []
        for item in items:
            y, y_mask, source = self._load_one_cpu(item["dataset"], item["sample_id"])
            ys.append(y)
            masks.append(y_mask)
            sources.append(source)
        y = torch.cat(ys, dim=0).to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
        y_mask = torch.cat(masks, dim=0).to(device=self.device, non_blocking=True)
        return y, y_mask, sources

    def stats(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "hits": self.hits,
            "fallbacks": self.fallbacks,
            "cache_items": len(self.cache),
            "text_cache_root": str(self.text_cache_root) if self.text_cache_root is not None else None,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--text-embedding", type=Path, default=DEFAULT_TEXT)
    ap.add_argument("--text-cache-root", type=Path, default=DEFAULT_TEXT_CACHE_ROOT)
    ap.add_argument("--text-cache-lru-items", type=int, default=128)
    ap.add_argument("--require-text-datasets", default="spatialvid")
    ap.add_argument("--run-name", default=f"v1-wan-stage1-t4-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--out-root", type=Path, default=ROOT / "log")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", choices=["latent_prefix", "dit_condition"], default="latent_prefix")
    ap.add_argument("--train-scope", choices=["full", "register"], default="full")
    ap.add_argument("--history-iw-chunks", default="1,4,8,16")
    ap.add_argument(
        "--dataset-weights",
        default="spatialvid=0.45,dl3dv=0.35,re10k=0.15,argoverse2=0.05",
    )
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--visual-flow-shift", type=float, default=7.0)
    ap.add_argument("--action-flow-shift", type=float, default=5.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    del args.num_workers  # payload-per-episode loading is intentionally explicit for now.
    cp_util.dp_rank = cp_util.cp_rank = 0
    cp_util.dp_size = cp_util.cp_size = 1
    install_non_reentrant_checkpoint()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    micro_frames = 13
    micro_stride = 12
    history_iw_cycle = parse_iw_history(args.history_iw_chunks)
    history_micro_by_iw = {
        iw: micro_steps_for_iw_chunks(iw, micro_frames=micro_frames, micro_stride=micro_stride)
        for iw in history_iw_cycle
    }
    dataset_weights = parse_dataset_weights(args.dataset_weights)
    require_text_datasets = parse_name_set(args.require_text_datasets)
    (
        window_buckets,
        window_counts_by_iw,
        window_counts_by_iw_dataset,
        files,
        filter_stats,
    ) = build_window_buckets_filtered(
        args.data_root,
        history_micro_by_iw,
        text_cache_root=args.text_cache_root,
        require_text_datasets=require_text_datasets,
    )
    if not files or not any(window_counts_by_iw.values()):
        raise ValueError(f"没有可用 T4 temporal windows: {args.data_root}")
    missing_history = [iw for iw, count in window_counts_by_iw.items() if count == 0]
    if missing_history:
        raise ValueError(f"这些 history 没有可用 windows: {missing_history}")

    device = torch.device(args.device)
    run_dir = args.out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(run_dir / "tensorboard")

    model = make_model(
        checkpoint=args.checkpoint,
        device=device,
        variant=args.variant,
        train_scope=args.train_scope,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = RFlowScheduler(
        shift=args.visual_flow_shift,
        use_reversed_velocity=True,
        use_timestep_transform=True,
    )
    text_conditions = TextConditionCache(
        empty_path=args.text_embedding,
        text_cache_root=args.text_cache_root,
        device=device,
        max_items=args.text_cache_lru_items,
    )

    rng = random.Random(args.seed)
    for by_dataset in window_buckets.values():
        for samples in by_dataset.values():
            rng.shuffle(samples)
    window_cursors = {
        iw: {dataset: 0 for dataset in by_dataset}
        for iw, by_dataset in window_buckets.items()
    }
    observed_dataset_counts = {}
    for path in files:
        observed_dataset_counts[path.parent.name] = observed_dataset_counts.get(path.parent.name, 0) + 1
    hist_cycle = list(history_iw_cycle)
    hist_cursor = 0

    def next_window_batch(iw_history: int) -> list[dict[str, Any]]:
        by_dataset = window_buckets[iw_history]
        available = [
            (dataset, dataset_weights.get(dataset, 0.0))
            for dataset, samples in by_dataset.items()
            if samples and dataset_weights.get(dataset, 0.0) > 0
        ]
        if not available:
            available = [
                (dataset, 1.0)
                for dataset, samples in by_dataset.items()
                if samples
            ]
        datasets = [item[0] for item in available]
        weights = [item[1] for item in available]
        dataset = rng.choices(datasets, weights=weights, k=1)[0]
        samples = by_dataset[dataset]
        cursor = window_cursors[iw_history][dataset]
        if cursor + args.batch_size > len(samples):
            rng.shuffle(samples)
            cursor = 0
        selected = samples[cursor : cursor + args.batch_size]
        window_cursors[iw_history][dataset] = cursor + args.batch_size
        return selected

    def next_iw_history() -> int:
        nonlocal hist_cursor
        if hist_cursor >= len(hist_cycle):
            rng.shuffle(hist_cycle)
            hist_cursor = 0
        value = hist_cycle[hist_cursor]
        hist_cursor += 1
        return value

    config = {
        "mode": "v1_stage1_t4_iw_aligned",
        "sample_unit": "temporal_window",
        "data_root": str(args.data_root),
        "num_eligible_episodes": len(files),
        "num_training_windows": sum(window_counts_by_iw.values()),
        "num_training_windows_by_iw": window_counts_by_iw,
        "num_training_windows_by_iw_dataset": window_counts_by_iw_dataset,
        "eligible_episodes_by_dataset": observed_dataset_counts,
        "data_filter": filter_stats,
        "dataset_weights": dataset_weights,
        "checkpoint": str(args.checkpoint),
        "checkpoint_audit": getattr(model, "_checkpoint_audit", {}),
        "variant": args.variant,
        "train_scope": args.train_scope,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "history_iw_chunks": history_iw_cycle,
        "history_micro_by_iw": history_micro_by_iw,
        "micro_frames": micro_frames,
        "micro_stride": micro_stride,
        "target_latent_t": 4,
        "target_rgb_frames": 13,
        "timestep_schema": {
            "source": "GigaWorld-Policy-style per-token timestep layout",
            "clean_prefix_history_local": 0,
            "future_visual": "visual_t sampled by RFlowScheduler",
            "a_query_action_tokens": "action_t sampled independently",
            "visual_flow_shift": args.visual_flow_shift,
            "action_flow_shift": args.action_flow_shift,
            "stage_one_action_loss": 0.0,
        },
        "text_condition": {
            "mode": "per-sample sidecar if present; otherwise empty UMT5 fallback",
            "empty_embedding": str(args.text_embedding),
            "text_cache_root": str(args.text_cache_root),
            "text_cache_lru_items": args.text_cache_lru_items,
            "require_text_datasets": sorted(require_text_datasets),
        },
        "history_action": "A_hist move/view embedding modulates each history chunk before Register extract/update",
        "current_action": (
            "A_cur move/view is encoded as noisy-future-only DiT condition for "
            "video generation; native InfiniteWorld action_encoder receives "
            "no-op to avoid double injection"
        ),
        "action_output": (
            "A_query tokens are inserted into the shared Wan/DiT token stream; "
            "A_query uses independent Giga-style action_timestep; Stage One "
            "computes no action supervision"
        ),
        "lambda_action": 0.0,
        "objective": "RFlow loss on T_latent=4 target; Register history span is IW-equivalent",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    log_path = run_dir / "train.log"
    with log_path.open("w") as log:
        log.write(json.dumps({"event": "start", **config}, ensure_ascii=False) + "\n")

        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            last_iw = None
            last_micro = None
            last_visual_t_mean = None
            last_action_t_mean = None
            for _ in range(args.gradient_accumulation_steps):
                iw_history = next_iw_history()
                window_batch = next_window_batch(iw_history)
                payloads = [load_payload(sample["path"]) for sample in window_batch]
                micro_history = int(window_batch[0]["history_micro"])
                last_iw, last_micro = iw_history, micro_history

                windows = []
                history_move_batches = [[] for _ in range(micro_history)]
                history_view_batches = [[] for _ in range(micro_history)]
                move_batch = []
                view_batch = []
                for item, sample in zip(payloads, window_batch):
                    chunks = item["micro_latents"]
                    start_micro = int(sample["start_micro"])
                    windows.append(chunks[:, start_micro : start_micro + micro_history + 1])
                    for history_offset in range(micro_history):
                        history_frame_start = (start_micro + history_offset) * micro_stride
                        history_move_batches[history_offset].append(
                            pad_action_81(item["move"], history_frame_start, micro_frames)
                        )
                        history_view_batches[history_offset].append(
                            pad_action_81(item["view"], history_frame_start, micro_frames)
                        )
                    target_frame_start = (start_micro + micro_history) * micro_stride
                    move_batch.append(pad_action_81(item["move"], target_frame_start, micro_frames))
                    view_batch.append(pad_action_81(item["view"], target_frame_start, micro_frames))

                chunks = torch.cat(windows, dim=0)
                histories = [
                    chunks[:, i].to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    for i in range(micro_history)
                ]
                target = chunks[:, -1].to(device=device, dtype=torch.bfloat16, non_blocking=True)
                history_moves = [
                    torch.stack(values).to(device=device, non_blocking=True)
                    for values in history_move_batches
                ]
                history_views = [
                    torch.stack(values).to(device=device, non_blocking=True)
                    for values in history_view_batches
                ]
                move = torch.stack(move_batch).to(device=device, non_blocking=True)
                view = torch.stack(view_batch).to(device=device, non_blocking=True)
                y_batch, y_mask_batch, _text_sources = text_conditions.batch(window_batch)

                conditioned_first = model.action_interface.condition_history_chunk(
                    histories[0], history_moves[0], history_views[0]
                )
                registers = model.register_memory.extract(conditioned_first)
                for history, history_move, history_view in zip(
                    histories[1:], history_moves[1:], history_views[1:]
                ):
                    history = model.action_interface.condition_history_chunk(
                        history, history_move, history_view
                    )
                    registers = model.register_memory.update(registers, history)
                local_latent = histories[-1][:, :, -1:]
                cond_t = getattr(model.register_memory, "register_frames", 0) if args.variant == "latent_prefix" else 0
                ignore_t = cond_t + local_latent.shape[2] + target.shape[2]
                ignore_mask = torch.zeros(
                    target.shape[0],
                    ignore_t,
                    target.shape[3],
                    target.shape[4],
                    device=device,
                    dtype=torch.bool,
                )
                visual_timestep = sample_visual_timestep(scheduler, target)
                action_timestep = sample_action_timestep(
                    target.shape[0],
                    device=device,
                    shift=args.action_flow_shift,
                    num_timesteps=scheduler.num_timesteps,
                )
                last_visual_t_mean = float(visual_timestep.detach().float().mean().cpu())
                last_action_t_mean = float(action_timestep.detach().float().mean().cpu())
                terms = scheduler.training_losses(
                    model,
                    target,
                    model_kwargs={
                        "y": y_batch,
                        "y_mask": y_mask_batch,
                        "registers": registers,
                        "local_latent": local_latent,
                        "move": move,
                        "view": view,
                        "action_timestep": action_timestep,
                    },
                    x_ignore_mask=ignore_mask,
                    t=visual_timestep,
                )
                loss = terms["loss"].mean()
                loss_sum += float(loss.detach().cpu())
                (loss / args.gradient_accumulation_steps).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_gib = torch.cuda.max_memory_reserved(device) / 2**30
            else:
                peak_gib = 0.0
            if step == 1 or step % args.log_every == 0:
                payload = {
                    "event": "step",
                    "step": step,
                    "loss": loss_sum / args.gradient_accumulation_steps,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "history_iw_chunks": last_iw,
                    "history_micro_steps": last_micro,
                    "target_latent_t": 4,
                    "visual_timestep_mean": last_visual_t_mean,
                    "action_timestep_mean": last_action_t_mean,
                    "peak_reserved_gib": peak_gib,
                    "text": text_conditions.stats(),
                }
                print(json.dumps(payload, ensure_ascii=False), flush=True)
                log.write(json.dumps(payload, ensure_ascii=False) + "\n")
                log.flush()
                writer.add_scalar("train/loss", payload["loss"], step)
                writer.add_scalar("train/grad_norm", payload["grad_norm"], step)
                writer.add_scalar("train/history_iw_chunks", float(last_iw or 0), step)
                writer.add_scalar("train/history_micro_steps", float(last_micro or 0), step)
                writer.add_scalar("timestep/visual_t_mean", float(last_visual_t_mean or 0), step)
                writer.add_scalar("timestep/action_t_mean", float(last_action_t_mean or 0), step)
                writer.add_scalar("system/peak_reserved_gib", peak_gib, step)
                writer.add_scalar("text/fallback_rate", (
                    text_conditions.fallbacks / max(1, text_conditions.requests)
                ), step)
            if step % args.save_every == 0:
                torch.save({
                    "backbone": model.backbone.state_dict(),
                    "register_memory": model.register_memory.state_dict(),
                    "action_interface": model.action_interface.state_dict(),
                    "shared_action": model.shared_action_state_dict(),
                    "step": step,
                    "variant": args.variant,
                    "config": config,
                }, run_dir / f"full-step-{step:06d}.pt")

        final_path = run_dir / "full-final.pt"
        torch.save({
            "backbone": model.backbone.state_dict(),
            "register_memory": model.register_memory.state_dict(),
            "action_interface": model.action_interface.state_dict(),
            "shared_action": model.shared_action_state_dict(),
            "step": args.steps,
            "variant": args.variant,
            "config": config,
        }, final_path)
        log.write(json.dumps({"event": "complete", "final_checkpoint": str(final_path)}, ensure_ascii=False) + "\n")
    writer.close()


if __name__ == "__main__":
    main()
