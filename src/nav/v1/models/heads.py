"""V1 generation and action heads."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    def __init__(self, *, hidden_dim: int, num_primitives: int = 12) -> None:
        super().__init__()
        self.primitive = nn.Linear(hidden_dim, num_primitives)
        self.delta = nn.Linear(hidden_dim, 6)

    def forward(self, action_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "primitive_logits": self.primitive(action_tokens),
            "delta_ego": self.delta(action_tokens),
        }


class FutureLatentHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        out_channels: int = 16,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        pt, ph, pw = patch_size
        self.proj = nn.Linear(hidden_dim, out_channels * pt * ph * pw)

    def forward(
        self,
        future_tokens: torch.Tensor,
        *,
        latent_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """tokens [B,N,D] -> latent patch prediction [B,C,T,H,W]."""

        t, h, w = latent_shape
        pt, ph, pw = self.patch_size
        b, n, _ = future_tokens.shape
        grid_t, grid_h, grid_w = t // pt, h // ph, w // pw
        expected = grid_t * grid_h * grid_w
        if n != expected:
            raise ValueError(f"future token count {n} != expected {expected}")
        patches = self.proj(future_tokens)
        patches = patches.view(b, grid_t, grid_h, grid_w, self.out_channels, pt, ph, pw)
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return patches.view(b, self.out_channels, t, h, w)


def _grid_shape(latent_shape: tuple[int, int, int], patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    t, h, w = latent_shape
    pt, ph, pw = patch_size
    if t % pt != 0 or h % ph != 0 or w % pw != 0:
        raise ValueError(f"latent_shape={latent_shape} is not divisible by patch_size={patch_size}")
    return t // pt, h // ph, w // pw


class FramePoseHead(nn.Module):
    """Read frame-level camera pose from current/local visual hidden states.

    Output format follows the project Stage Two contract:
    ``[tx, ty, tz, qw, qx, qy, qz, fov_x, fov_y]`` per latent frame.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        out_dim: int = 9,
        variant: Literal["mlp", "linear"] = "mlp",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.variant = variant
        if variant == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        elif variant == "linear":
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise ValueError(f"unsupported FramePoseHead variant: {variant}")

    def forward(self, obs_tokens: torch.Tensor, *, latent_shape: tuple[int, int, int]) -> torch.Tensor:
        grid_t, grid_h, grid_w = _grid_shape(latent_shape, self.patch_size)
        b, n, d = obs_tokens.shape
        expected = grid_t * grid_h * grid_w
        if n != expected:
            raise ValueError(f"obs token count {n} != expected {expected}")
        grid = obs_tokens.view(b, grid_t, grid_h, grid_w, d)
        pooled = grid.mean(dim=(2, 3))
        return self.net(pooled)


class FrameDepthHead(nn.Module):
    """Low-resolution dense depth probe from current/local visual hidden states.

    The head predicts one depth grid per latent frame at visual-token grid
    resolution.  Training scripts may downsample pseudo/GT depth to this grid.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.depth = nn.Linear(hidden_dim, 1)
        self.conf = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs_tokens: torch.Tensor,
        *,
        latent_shape: tuple[int, int, int],
    ) -> dict[str, torch.Tensor]:
        grid_t, grid_h, grid_w = _grid_shape(latent_shape, self.patch_size)
        b, n, d = obs_tokens.shape
        expected = grid_t * grid_h * grid_w
        if n != expected:
            raise ValueError(f"obs token count {n} != expected {expected}")
        hidden = self.proj(obs_tokens)
        depth = F.softplus(self.depth(hidden).squeeze(-1)) + 1e-4
        confidence = 1.0 + F.softplus(self.conf(hidden).squeeze(-1))
        return {
            "depth": depth.view(b, grid_t, grid_h, grid_w),
            "depth_conf": confidence.view(b, grid_t, grid_h, grid_w),
        }
