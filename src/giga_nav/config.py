"""Configuration for the GigaWorld-style navigation ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GigaNavConfig:
    """Native Wan2.1 geometry plus Giga-style policy token settings.

    ``action_horizon=48`` follows GigaWorld-Policy's p=48 setting.  R2R
    samples are padded with terminal STOP labels by the existing canonical
    loader; ``--action-horizon`` can be lowered for a controlled navigation
    ablation without changing the backbone.
    """

    backbone_checkpoint: Path = Path("/sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    latent_channels: int = 16
    latent_frames: int = 4
    latent_height: int = 56
    latent_width: int = 112
    patch_size: tuple[int, int, int] = (1, 2, 2)
    hidden_dim: int = 1536
    ffn_dim: int = 8960
    freq_dim: int = 256
    num_heads: int = 12
    num_layers: int = 30
    caption_channels: int = 4096
    text_length: int = 512
    action_input_dim: int = 14
    state_input_dim: int = 14
    action_horizon: int = 48
    num_nav_classes: int = 4
    action_mlp_dim: int = 256
    state_tokens: int = 1
    use_zero_future_visual: bool = True
    train_backbone: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backbone_checkpoint"] = str(self.backbone_checkpoint)
        payload["patch_size"] = list(self.patch_size)
        return payload
