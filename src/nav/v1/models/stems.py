"""V1 visual/register/action stem modules."""

from __future__ import annotations

import torch
from torch import nn


class VisualPatchStem(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 16,
        hidden_dim: int = 1536,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        pt, ph, pw = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent [B,C,T,H,W] -> tokens [B,N,D]."""

        hidden = self.proj(latent)
        return hidden.flatten(2).transpose(1, 2).contiguous()


class RegisterStem(nn.Module):
    def __init__(self, *, register_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(register_dim, hidden_dim)

    def forward(self, register: torch.Tensor) -> torch.Tensor:
        return self.proj(register)


class ActionStem(nn.Module):
    def __init__(
        self,
        *,
        num_primitives: int = 12,
        hidden_dim: int = 1536,
        max_horizon: int = 64,
    ) -> None:
        super().__init__()
        self.primitive_embed = nn.Embedding(num_primitives, hidden_dim)
        self.query_embed = nn.Embedding(max_horizon, hidden_dim)
        self.timestep_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def query(self, batch_size: int, horizon: int, *, device: torch.device | str) -> torch.Tensor:
        positions = torch.arange(horizon, device=device)
        queries = self.query_embed(positions)[None].expand(batch_size, -1, -1)
        return queries.contiguous()

    def from_primitives(self, primitive_ids: torch.Tensor) -> torch.Tensor:
        return self.primitive_embed(primitive_ids.long())
