"""V1 generation and action heads."""

from __future__ import annotations

import torch
from torch import nn


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
