"""V1 Stage One scaffold model.

This module is a lightweight, Wan-compatible shape scaffold: it uses the same
V1 sparse latent tensors and policy-safe token layout, but a small Transformer
backbone instead of the full Wan/InfiniteWorld DiT. Its job is to validate the
new data/loss/input contract before the large backbone integration.
"""

from __future__ import annotations

import torch
from torch import nn

from .heads import ActionHead, FutureLatentHead
from .masks import V1TokenLayout, build_policy_safe_attention_mask
from .stems import ActionStem, VisualPatchStem


class RegisterExtractor(nn.Module):
    """Extract R0 from current observation tokens using learnable queries."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_register_tokens: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_register_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        batch = obs_tokens.shape[0]
        query = self.query[None].expand(batch, -1, -1)
        extracted, _ = self.cross_attn(query, obs_tokens, obs_tokens, need_weights=False)
        hidden = self.norm(query + extracted)
        return self.out_norm(hidden + self.mlp(hidden))


class V1StageOneScaffold(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 64,
        action_horizon: int = 4,
        num_primitives: int = 12,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_register_tokens = num_register_tokens
        self.action_horizon = action_horizon

        self.visual_stem = VisualPatchStem(hidden_dim=hidden_dim, patch_size=patch_size)
        self.action_stem = ActionStem(
            num_primitives=num_primitives,
            hidden_dim=hidden_dim,
            max_horizon=max(action_horizon, 64),
        )
        self.register_extractor = RegisterExtractor(
            hidden_dim=hidden_dim,
            num_register_tokens=num_register_tokens,
            num_heads=num_heads,
        )
        self.visual_timestep = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_type = nn.Embedding(4, hidden_dim)  # register, obs, future, action

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.future_head = FutureLatentHead(hidden_dim=hidden_dim, patch_size=patch_size)
        self.action_head = ActionHead(hidden_dim=hidden_dim, num_primitives=num_primitives)

    def forward(
        self,
        *,
        z_obs: torch.Tensor,
        z_future_noisy: torch.Tensor,
        visual_timestep: torch.Tensor,
    ) -> dict[str, torch.Tensor | V1TokenLayout]:
        batch = z_obs.shape[0]
        device = z_obs.device
        obs_tokens = self.visual_stem(z_obs)
        future_tokens = self.visual_stem(z_future_noisy)
        register_tokens = self.register_extractor(obs_tokens)
        action_tokens = self.action_stem.query(
            batch,
            self.action_horizon,
            device=device,
        )

        timestep = visual_timestep.to(dtype=future_tokens.dtype).view(batch, 1, 1)
        future_tokens = future_tokens + self.visual_timestep(timestep).to(future_tokens.dtype)

        pieces = [register_tokens, obs_tokens, future_tokens, action_tokens]
        type_ids = [
            torch.zeros(register_tokens.shape[1], dtype=torch.long, device=device),
            torch.ones(obs_tokens.shape[1], dtype=torch.long, device=device),
            torch.full((future_tokens.shape[1],), 2, dtype=torch.long, device=device),
            torch.full((action_tokens.shape[1],), 3, dtype=torch.long, device=device),
        ]
        tokens = torch.cat(pieces, dim=1)
        token_type = torch.cat(type_ids, dim=0)
        tokens = tokens + self.token_type(token_type)[None].to(tokens.dtype)

        layout = V1TokenLayout(
            n_register=register_tokens.shape[1],
            n_obs=obs_tokens.shape[1],
            n_future=future_tokens.shape[1],
            n_action=action_tokens.shape[1],
        )
        allowed = build_policy_safe_attention_mask(layout, device=device)
        src_mask = ~allowed
        hidden = self.backbone(tokens, mask=src_mask)
        hidden = self.final_norm(hidden)

        future_hidden = hidden[:, layout.future, :]
        action_hidden = hidden[:, layout.action, :]
        pred_noise = self.future_head(
            future_hidden,
            latent_shape=(
                z_future_noisy.shape[2],
                z_future_noisy.shape[3],
                z_future_noisy.shape[4],
            ),
        )
        action_out = self.action_head(action_hidden)
        return {
            "pred_noise": pred_noise,
            "primitive_logits": action_out["primitive_logits"],
            "delta_ego": action_out["delta_ego"],
            "layout": layout,
        }
