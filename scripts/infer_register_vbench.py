#!/usr/bin/env python3
"""按 InfiniteWorld 本地 VBench 协议生成 A/B 对比视频。"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch import nn

NAV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = NAV_ROOT.parent
INFINITE_ROOT = REPO_ROOT / "Infinite-World"
sys.path.insert(0, str(NAV_ROOT / "src"))
sys.path.insert(0, str(INFINITE_ROOT))

import infworld.context_parallel.context_parallel_util as cp_util
from infworld.configs import bucket_config as bucket_config_module
from infworld.utils.data_utils import save_silent_video
from infworld.utils.prepare_dataloader import get_obj_from_str
from nav.infinite_adapter import InfiniteRegisterAdapter


MOVE_ACTION_MAP = {
    "no-op": 0, "go forward": 1, "go back": 2, "go left": 3,
    "go right": 4, "go forward and go left": 5,
    "go forward and go right": 6, "go back and go left": 7,
    "go back and go right": 8, "uncertain": 9,
}
VIEW_ACTION_MAP = {
    "no-op": 0, "turn up": 1, "turn down": 2, "turn left": 3,
    "turn right": 4, "turn up and turn left": 5,
    "turn up and turn right": 6, "turn down and turn left": 7,
    "turn down and turn right": 8, "uncertain": 9,
}
NEGATIVE_PROMPT = (
    "many cars, crowds, Vivid hues, overexposed, static, blurry details, "
    "subtitles, style, work, artwork, image, still, overall grayish, worst "
    "quality, low quality, JPEG compression artifacts, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn face, deformed, "
    "disfigured, deformed limbs, fused fingers, motionless image, cluttered "
    "background, three legs, crowded background, walking backwards."
)


class CFGRegisterWrapper(nn.Module):
    """消费 scheduler 的兼容字段，并把 A/B 条件准确转发到 NAV 模型。"""

    def __init__(self, model: InfiniteRegisterAdapter) -> None:
        super().__init__()
        self.model = model

    @property
    def y_embedder(self):
        return self.model.backbone.y_embedder

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        image_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # RFlowScheduler 强制要求并复制 image_cond；A/B 已彻底删除 HPMC，
        # 因此这里只消费该兼容字段，真正 history 来自 registers。
        del image_cond
        return self.model(x=x, t=t, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=["latent_prefix", "dit_condition"], required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="每个 prompt 分别生成这些随机种子",
    )
    parser.add_argument(
        "--prompt-indices", type=int, nargs="+", default=[0],
        help="demo.yaml 中要生成的 prompt 下标",
    )
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_condition_image(path: Path, bucket_config: dict) -> torch.Tensor:
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    ratio = image.shape[0] / image.shape[1]
    closest = min(bucket_config, key=lambda x: abs(float(x) - ratio))
    target_h, target_w = bucket_config[closest][0]
    scale = max(target_h / image.shape[0], target_w / image.shape[1])
    resized = cv2.resize(
        image,
        (math.ceil(scale * image.shape[1]), math.ceil(scale * image.shape[0])),
        interpolation=cv2.INTER_AREA,
    )
    tensor = torch.from_numpy(resized)[None].permute(0, 3, 1, 2).contiguous()
    tensor = transforms.functional.center_crop(tensor, (target_h, target_w))
    return ((tensor[:, :, None] / 255.0 - 0.5) * 2).float()


def load_actions(path: Path) -> tuple[list[int], list[int]]:
    actions = json.loads(path.read_text())
    return (
        [MOVE_ACTION_MAP[x["move"]] for x in actions],
        [VIEW_ACTION_MAP[x["view"]] for x in actions],
    )


def build_model(
    variant: str, checkpoint: Path, device: torch.device
) -> InfiniteRegisterAdapter:
    model = InfiniteRegisterAdapter(
        variant=variant,
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
            if variant == "latent_prefix"
            else {
                "latent_channels": 16, "register_dim": 256,
                "num_registers": 16, "num_update_layers": 2,
                "num_heads": 8, "caption_channels": 4096,
            }
        ),
    )
    model.remove_hmpc()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("variant") != variant:
        raise ValueError(
            f"权重 variant={state.get('variant')}，命令 variant={variant}"
        )
    model.backbone.load_state_dict(state["backbone"], strict=True)
    model.register_memory.load_state_dict(state["register_memory"], strict=True)
    print(f"[NAV] 已加载 {checkpoint}，step={state.get('step')}")
    return model.to(device=device, dtype=torch.bfloat16).eval()


def main() -> None:
    args = parse_args()
    setup_seed(args.seeds[0])
    cp_util.dp_rank = cp_util.cp_rank = 0
    cp_util.dp_size = cp_util.cp_size = 1
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(INFINITE_ROOT / "configs/infworld_config.yaml")
    vae = get_obj_from_str(config.vae_target)(**config.vae_cfg).to(device)
    text_encoder = get_obj_from_str(config.text_encoder_target)(
        device=device, **config.text_encoder_cfg
    )
    text_encoder.t5.model.to(device)
    scheduler = get_obj_from_str(config.scheduler_target)(
        **config.val_scheduler_cfg
    )
    scheduler.num_sampling_steps = args.steps
    scheduler.shift = 7
    model = build_model(args.variant, args.checkpoint, device)
    cfg_model = CFGRegisterWrapper(model)

    bucket_config = getattr(bucket_config_module, "ASPECT_RATIO_627_F64")
    all_prompts = OmegaConf.load(
        INFINITE_ROOT / "prompts/demo.yaml"
    ).prompts
    for prompt_index in args.prompt_indices:
        prompt, image_rel, action_rel = all_prompts[prompt_index]
        image_path = (INFINITE_ROOT / image_rel).resolve()
        action_path = (INFINITE_ROOT / action_rel).resolve()
        cond_video = load_condition_image(image_path, bucket_config).to(device)
        move_indices, view_indices = load_actions(action_path)

        for seed in args.seeds:
            video_name = (
                f"{prompt_index:04d}_seed{seed:04d}_"
                f"{prompt[:30].replace(' ', '_')}"
            )
            video_path = args.output_dir / (video_name + ".mp4")
            if video_path.exists():
                print(f"[NAV] 已存在，跳过：{video_path}")
                continue
            setup_seed(seed)
            with torch.inference_mode():
                history_latent = vae.encode(cond_video)
                video_buffer = cond_video.cpu()
                latent_size = torch.Size(
                    [history_latent.shape[0], history_latent.shape[1], 21,
                     history_latent.shape[3], history_latent.shape[4]]
                )
                registers = None

                for chunk_index in range(args.chunks):
                    # 第一个已知 Chunk 提取 Register；后续生成 Chunk 才更新。
                    # local memory 始终是最近帧。
                    registers = (
                        model.register_memory.extract(history_latent)
                        if registers is None
                        else model.register_memory.update(
                            registers, history_latent
                        )
                    )
                    local_latent = history_latent[:, :, -1:]
                    start = video_buffer.shape[2] - 1
                    end = start + int(config.validation_data.num_frames)
                    move = torch.tensor(
                        move_indices[start:end],
                        device=device,
                        dtype=torch.long,
                    )
                    view = torch.tensor(
                        view_indices[start:end],
                        device=device,
                        dtype=torch.long,
                    )
                    if move.numel() < int(config.validation_data.num_frames):
                        pad = (
                            int(config.validation_data.num_frames)
                            - move.numel()
                        )
                        move = torch.cat([move, move.new_zeros(pad)])
                        view = torch.cat([view, view.new_zeros(pad)])

                    # CFG 将 positive/negative prompt 合成 batch=2。
                    additional_args = {
                        "image_cond": history_latent,
                        "registers": registers.repeat(
                            2, *([1] * (registers.ndim - 1))
                        ),
                        "local_latent": local_latent.repeat(
                            2, 1, 1, 1, 1
                        ),
                        "move": move[None].repeat(2, 1),
                        "view": view[None].repeat(2, 1),
                    }
                    print(
                        f"[NAV] prompt={prompt_index} seed={seed}，"
                        f"生成 chunk {chunk_index + 1}/{args.chunks}"
                    )
                    samples = scheduler.sample(
                        model=cfg_model,
                        text_encoder=text_encoder,
                        null_embedder=cfg_model.y_embedder,
                        z_size=latent_size,
                        prompts=[prompt],
                        guidance_scale=args.cfg_scale,
                        negative_prompts=[NEGATIVE_PROMPT],
                        device=device,
                        additional_args=additional_args,
                    )
                    # 与 GPU 0 上的 VGGT 共存时显存余量很小。采样结束后
                    # 主动归还 DiT 的缓存块，避免 VAE decode 因碎片无法取得
                    # 不到 1 GiB 的连续空间。
                    torch.cuda.empty_cache()
                    decoded = vae.decode(samples).cpu()
                    video_buffer = torch.cat(
                        [video_buffer, decoded[:, :, 1:]], dim=2
                    )
                    history_latent = samples
                    torch.cuda.empty_cache()

            save_silent_video(
                video_buffer.to(device),
                str(args.output_dir / video_name),
                fps=30,
                quality=10,
            )
            print(
                f"[NAV] 完成：{args.output_dir / (video_name + '.mp4')}，"
                f"{video_buffer.shape[2]} frames"
            )


if __name__ == "__main__":
    main()
