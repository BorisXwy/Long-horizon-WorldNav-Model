#!/usr/bin/env python3
"""在 Infinite-World diffusion loss 上训练 Register adapter。

默认冻结 1.3B backbone，只更新 Register 模块；这是当前两张 48GB GPU 上可行的
第一阶段。输入为 cache_re10k_latents.py 生成的真实 RE10K latent。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "Infinite-World"))

import infworld.context_parallel.context_parallel_util as cp_util
import infworld.models.dit_model as dit_model_module
from infworld.models.checkpoint import set_grad_checkpoint
from infworld.models.scheduler import RFlowScheduler
from nav.infinite_adapter import InfiniteRegisterAdapter


def install_non_reentrant_checkpoint() -> None:
    """修复上游 reentrant checkpoint 与 block history cache 的冲突。"""

    def checkpoint_module(module, *args, **kwargs):
        if getattr(module, "grad_checkpointing", False):
            return torch.utils.checkpoint.checkpoint(
                module, *args, use_reentrant=False, **kwargs
            )
        return module(*args, **kwargs)

    # WanModel.forward 引用的是 dit_model.py 模块级符号。
    dit_model_module.auto_grad_checkpoint = checkpoint_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["latent_prefix", "dit_condition"], required=True)
    parser.add_argument("--latent-cache", type=Path, required=True,
                        help="单个 .pt 文件或包含 .pt 文件的目录")
    parser.add_argument(
        "--action-cache",
        type=Path,
        help="当 latent payload 不含 move/view 时，按 sample_id 读取 JSON",
    )
    parser.add_argument(
        "--cache-manifest", type=Path,
        default=Path("/sharedata/NAV/derived/manifests/episodes.jsonl"),
        help="用于在不读取大 tensor 的情况下筛选 num_chunks",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--min-chunks", type=int, default=2)
    parser.add_argument(
        "--max-chunks", type=int, default=0,
        help="0 表示不设上限；每个 batch 随机选择范围内的连续 chunk 窗口",
    )
    parser.add_argument(
        "--history-lengths",
        default="",
        help="逗号分隔的 history chunk 数；设置后按打乱循环等比例采样",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260727)
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="从上一 curriculum stage 的完整 backbone+register 权重继续",
    )
    parser.add_argument("--plateau-window", type=int, default=200)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--min-steps-before-plateau", type=int, default=1000)
    parser.add_argument(
        "--train-scope",
        choices=["register", "full"],
        default="register",
        help="register 只训练新增模块；full 训练移除 HPMC 后的全部参数",
    )
    parser.add_argument(
        "--register-checkpoint",
        type=Path,
        help="继续使用 Register 阶段权重；全参训练建议指定",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--text-embedding",
        type=Path,
        default=Path(
            "/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt"
        ),
    )
    args = parser.parse_args()
    history_lengths = [
        int(value) for value in args.history_lengths.split(",") if value
    ]
    if any(value < 1 for value in history_lengths):
        raise ValueError("--history-lengths 必须均为正整数")

    cp_util.dp_rank = cp_util.cp_rank = 0
    cp_util.dp_size = cp_util.cp_size = 1
    torch.manual_seed(42)
    install_non_reentrant_checkpoint()
    device = torch.device(args.device)
    run_name = args.run_name or f"full-{args.variant}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = ROOT / "log" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(run_dir / "tensorboard")

    model = InfiniteRegisterAdapter(
        variant=args.variant,
        model_cfg={
            "model_type": "t2v", "dim": 1536, "in_channels": 20,
            "ffn_dim": 8960, "freq_dim": 256, "num_heads": 12,
            "num_layers": 30, "out_channels": 16,
            "caption_channels": 4096, "model_max_length": 512,
        },
        register_cfg=(
            {
                "channels": 16, "register_frames": 4, "hidden_dim": 64,
                "num_heads": 4, "num_update_layers": 2,
                "chunk_time_tokens": 8,
            }
            if args.variant == "latent_prefix"
            else {
                "latent_channels": 16, "register_dim": 256,
                "num_registers": 16, "num_update_layers": 2,
                "num_heads": 8, "caption_channels": 4096,
            }
        ),
    )
    audit = model.load_infinite_checkpoint(
        "/sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt"
    )
    if args.resume_checkpoint:
        resume = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if resume.get("variant") != args.variant:
            raise ValueError(
                f"resume variant={resume.get('variant')} != {args.variant}"
            )
        model.backbone.load_state_dict(resume["backbone"], strict=True)
        model.register_memory.load_state_dict(
            resume["register_memory"], strict=True
        )
    if args.register_checkpoint:
        register_state = torch.load(
            args.register_checkpoint, map_location="cpu", weights_only=False
        )
        model.register_memory.load_state_dict(register_state, strict=True)
    if args.train_scope == "register":
        model.freeze_backbone()
    else:
        model.requires_grad_(True)
    model.to(device=device, dtype=torch.bfloat16).train()
    # 冻结参数仍需对 Register 输入求梯度，checkpoint 显著降低激活显存。
    set_grad_checkpoint(model.backbone)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=1e-5, weight_decay=0.01
    )
    scheduler = RFlowScheduler(
        shift=7.0, use_reversed_velocity=True, use_timestep_transform=True
    )
    cache_files = (
        sorted(args.latent_cache.rglob("*.pt"))
        if args.latent_cache.is_dir()
        else [args.latent_cache]
    )
    if not cache_files:
        raise FileNotFoundError(f"没有 latent cache: {args.latent_cache}")
    if not args.text_embedding.exists():
        raise FileNotFoundError(
            f"缺少文本 embedding: {args.text_embedding}；"
            "先运行 scripts/cache_empty_text.py"
        )
    text_condition = torch.load(
        args.text_embedding, map_location="cpu", weights_only=False
    )
    y = text_condition["y"].to(device=device, dtype=torch.bfloat16)
    y_mask = text_condition["y_mask"].to(device=device)
    chunk_index = {}
    if args.cache_manifest.is_file():
        with args.cache_manifest.open() as stream:
            for line in stream:
                item = json.loads(line)
                chunk_index[item["sample_id"]] = int(item["num_chunks"])
    eligible = []
    required_min_chunks = (
        max(history_lengths) + 1 if history_lengths else args.min_chunks
    )
    for path in cache_files:
        sidecar = path.with_suffix(".json")
        if sidecar.is_file():
            num_chunks = int(
                json.loads(sidecar.read_text())["cached_chunks"]
            )
        else:
            num_chunks = chunk_index.get(path.stem)
        if num_chunks is None:
            metadata = torch.load(
                path, map_location="cpu", weights_only=False
            )
            num_chunks = int(
                metadata.get("num_chunks", metadata["chunks"].shape[1])
            )
        if num_chunks >= required_min_chunks:
            eligible.append(path)
    cache_files = eligible
    if not cache_files:
        raise ValueError(
            f"没有 num_chunks >= {required_min_chunks} 的缓存样本"
        )
    shuffle_rng = random.Random(args.shuffle_seed)
    shuffled_files = list(cache_files)
    shuffle_rng.shuffle(shuffled_files)
    sample_cursor = 0
    shuffle_epoch = 0
    history_length_cycle = list(history_lengths)
    history_length_cursor = 0
    if history_length_cycle:
        shuffle_rng.shuffle(history_length_cycle)

    def next_batch_files() -> list[Path]:
        nonlocal sample_cursor, shuffle_epoch, shuffled_files
        if sample_cursor + args.batch_size > len(shuffled_files):
            shuffle_epoch += 1
            shuffle_rng.shuffle(shuffled_files)
            sample_cursor = 0
        selected = shuffled_files[
            sample_cursor : sample_cursor + args.batch_size
        ]
        sample_cursor += args.batch_size
        return selected

    def next_selected_chunks(available: int) -> int:
        nonlocal history_length_cursor, history_length_cycle
        if history_length_cycle:
            if history_length_cursor >= len(history_length_cycle):
                shuffle_rng.shuffle(history_length_cycle)
                history_length_cursor = 0
            history_chunks = history_length_cycle[history_length_cursor]
            history_length_cursor += 1
            selected = history_chunks + 1
            if selected > available:
                raise RuntimeError(
                    f"样本只有{available} chunks，无法采样{history_chunks} history"
                )
            return selected
        upper = (
            available
            if args.max_chunks == 0
            else min(args.max_chunks, available)
        )
        if upper < args.min_chunks:
            raise RuntimeError("shuffle 后出现不满足 chunk 范围的样本")
        return shuffle_rng.randint(args.min_chunks, upper)

    log_path = run_dir / "train.log"
    with log_path.open("w") as log:
        log.write(json.dumps({
            "checkpoint_audit": audit,
            "source": str(args.latent_cache),
            "action_cache": (
                str(args.action_cache) if args.action_cache else None
            ),
            "num_cached_samples": len(cache_files),
            "shuffle": True,
            "shuffle_seed": args.shuffle_seed,
            "min_chunks": args.min_chunks,
            "max_chunks": args.max_chunks,
            "history_lengths": history_lengths,
            "history_sampling": (
                "equal_shuffled_cycle" if history_lengths else "uniform_range"
            ),
            "train_scope": args.train_scope,
            "trainable_parameters": sum(p.numel() for p in trainable_parameters),
            "register_checkpoint": (
                str(args.register_checkpoint) if args.register_checkpoint else None
            ),
            "resume_checkpoint": (
                str(args.resume_checkpoint) if args.resume_checkpoint else None
            ),
            "text_embedding": str(args.text_embedding),
            "text": text_condition.get("text", ""),
            "micro_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
        }) + "\n")
        recent_losses: deque[float] = deque(maxlen=args.plateau_window)
        plateau_reference = None
        plateau_bad_windows = 0
        final_step = 0
        for step in range(1, args.steps + 1):
            final_step = step
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for accumulation_index in range(
                args.gradient_accumulation_steps
            ):
                batch_files = next_batch_files()
                cached_batch = [
                    torch.load(p, map_location="cpu", weights_only=False)
                    for p in batch_files
                ]
                if args.action_cache:
                    for path, item in zip(batch_files, cached_batch):
                        if "move" in item and "view" in item:
                            continue
                        action_path = args.action_cache / f"{path.stem}.json"
                        if not action_path.is_file():
                            raise FileNotFoundError(
                                f"缺少 action sidecar：{action_path}"
                            )
                        action = json.loads(action_path.read_text())
                        item["move"] = torch.tensor(
                            action["move"], dtype=torch.long
                        )
                        item["view"] = torch.tensor(
                            action["view"], dtype=torch.long
                        )
                available = min(
                    int(x.get("num_chunks", x["chunks"].shape[1]))
                    for x in cached_batch
                )
                selected_chunks = next_selected_chunks(available)
                windows = []
                move_items = []
                view_items = []
                for item in cached_batch:
                    item_chunks = int(
                        item.get("num_chunks", item["chunks"].shape[1])
                    )
                    start_chunk = shuffle_rng.randint(
                        0, item_chunks - selected_chunks
                    )
                    windows.append(
                        item["chunks"][
                            :, start_chunk : start_chunk + selected_chunks
                        ]
                    )
                    action_start = (
                        start_chunk + selected_chunks - 1
                    ) * 81
                    move_items.append(
                        item["move"][action_start : action_start + 81]
                    )
                    view_items.append(
                        item["view"][action_start : action_start + 81]
                    )
                chunks = torch.cat(windows, dim=0)
                histories = [
                    chunks[:, index].to(
                        device=device, dtype=torch.bfloat16
                    )
                    for index in range(selected_chunks - 1)
                ]
                target = chunks[:, -1].to(
                    device=device, dtype=torch.bfloat16
                )
                move = torch.stack(move_items).to(device)
                view = torch.stack(view_items).to(device)
                ignore_mask = torch.zeros(
                    args.batch_size,
                    target.shape[2],
                    target.shape[3],
                    target.shape[4],
                    device=device,
                    dtype=torch.bool,
                )
                registers = model.register_memory.extract(histories[0])
                for history in histories[1:]:
                    registers = model.register_memory.update(
                        registers, history
                    )
                local = histories[-1][:, :, -1:]
                terms = scheduler.training_losses(
                    model,
                    target,
                    model_kwargs={
                        "y": y.expand(args.batch_size, -1, -1, -1),
                        "y_mask": y_mask.expand(args.batch_size, -1),
                        "registers": registers,
                        "local_latent": local,
                        "move": move,
                        "view": view,
                    },
                    x_ignore_mask=ignore_mask,
                )
                loss = terms["loss"].mean()
                accumulated_loss += float(loss.detach())
                (
                    loss / args.gradient_accumulation_steps
                ).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
            payload = {
                "step": step,
                "loss": (
                    accumulated_loss / args.gradient_accumulation_steps
                ),
                "register_grad_norm": float(grad_norm), "peak_gpu_gib": peak_gib,
                "variant": args.variant, "objective": "Infinite-World RFlow diffusion loss",
                "train_scope": args.train_scope,
                "micro_batch_size": args.batch_size,
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
                "effective_batch_size": (
                    args.batch_size
                    * args.gradient_accumulation_steps
                ),
                "selected_chunks": selected_chunks,
                "history_chunks": selected_chunks - 1,
                "shuffle_epoch": shuffle_epoch,
            }
            print(json.dumps(payload), flush=True)
            log.write(json.dumps(payload) + "\n")
            log.flush()
            writer.add_scalar("train/diffusion_loss", payload["loss"], step)
            writer.add_scalar("train/register_grad_norm", payload["register_grad_norm"], step)
            writer.add_scalar("system/peak_gpu_gib", peak_gib, step)
            writer.add_scalar("train/selected_chunks", selected_chunks, step)
            if step % args.save_every == 0:
                if args.train_scope == "full":
                    torch.save(
                        {
                            "backbone": model.backbone.state_dict(),
                            "register_memory": model.register_memory.state_dict(),
                            "step": step,
                            "variant": args.variant,
                        },
                        run_dir / f"full-step-{step:06d}.pt",
                    )
                else:
                    torch.save(
                        model.register_memory.state_dict(),
                        run_dir / f"register-step-{step:06d}.pt",
                    )
            recent_losses.append(payload["loss"])
            if (
                args.plateau_window > 0
                and step >= args.min_steps_before_plateau
                and len(recent_losses) == args.plateau_window
                and step % args.plateau_window == 0
            ):
                current = sum(recent_losses) / len(recent_losses)
                if plateau_reference is None:
                    plateau_reference = current
                else:
                    improvement = (
                        plateau_reference - current
                    ) / max(abs(plateau_reference), 1e-8)
                    if improvement < args.plateau_min_relative_improvement:
                        plateau_bad_windows += 1
                    else:
                        plateau_bad_windows = 0
                    plateau_reference = min(plateau_reference, current)
                    if plateau_bad_windows >= args.plateau_patience:
                        log.write(json.dumps({
                            "event": "plateau_stop",
                            "step": step,
                            "window_loss": current,
                        }) + "\n")
                        break
        if args.train_scope == "full":
            final_path = run_dir / "full-final.pt"
            torch.save(
                {
                    "backbone": model.backbone.state_dict(),
                    "register_memory": model.register_memory.state_dict(),
                    "step": final_step,
                    "variant": args.variant,
                },
                final_path,
            )
        else:
            final_path = run_dir / "register-final.pt"
            torch.save(model.register_memory.state_dict(), final_path)
        log.write(json.dumps({
            "event": "training_complete",
            "step": final_step,
            "final_checkpoint": str(final_path),
        }) + "\n")
    writer.close()
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
