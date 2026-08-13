"""Stage One history-action interface.

本模块只负责 `A_hist`：历史已执行 action / pseudo motion 调制 history chunk，
然后交给 Register Extractor / Updater。正式 V1 的 `A_cur`、`A_query` 和
`A_out` 必须进入 shared Wan/DiT backbone，由 `InfiniteRegisterAdapter` 管理。
这里不得再放 `Register/local -> 旁路 action head`。
"""

from __future__ import annotations

import torch
from torch import nn

class StageOneActionInterface(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int = 16,
        hidden_dim: int = 256,
        action_horizon: int = 4,
        vocab_size: int = 10,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.vocab_size = vocab_size

        self.move_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.view_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.history_to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, latent_channels),
        )

    def summarize_action(self, move: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
        move = move.clamp_min(0).clamp_max(self.vocab_size - 1)
        view = view.clamp_min(0).clamp_max(self.vocab_size - 1)
        move_hidden = self.move_embedding(move).mean(dim=1)
        view_hidden = self.view_embedding(view).mean(dim=1)
        return torch.cat([move_hidden, view_hidden], dim=-1)

    def condition_history_chunk(
        self,
        chunk_latent: torch.Tensor,
        move: torch.Tensor,
        view: torch.Tensor,
    ) -> torch.Tensor:
        """Inject previous executed action/motion into history chunk latent.

        The result is consumed by Register extract/update.  For datasets without
        action/motion, move/view are deterministic EMPTY/no-op labels, so the
        interface remains fixed.
        """

        summary = self.summarize_action(move.to(chunk_latent.device), view.to(chunk_latent.device))
        bias = self.history_to_latent(summary).to(chunk_latent.dtype)
        return chunk_latent + bias[:, :, None, None, None]
