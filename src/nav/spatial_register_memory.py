"""A 方案：固定 T=4、具有 latent-like 空间布局的 Register state。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialRegisterUpdateBlock(nn.Module):
    """每个空间位置独立地以 Register 查询该位置的 chunk 时间序列。"""

    def __init__(self, channels: int = 16, dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.register_in = nn.Linear(channels, dim)
        self.chunk_in = nn.Linear(channels, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=0.0
        )
        self.out = nn.Linear(dim, channels)
        self.channel_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(
        self, registers: torch.Tensor, chunk_latent: torch.Tensor
    ) -> torch.Tensor:
        # [B,C,T,H,W] -> [B*H*W,T,C]。空间坐标不混合，仅在同一位置做时间交互。
        b, c, register_t, h, w = registers.shape
        chunk_t = chunk_latent.shape[2]
        query = registers.permute(0, 3, 4, 2, 1).reshape(b * h * w, register_t, c)
        key_value = (
            chunk_latent.permute(0, 3, 4, 2, 1)
            .reshape(b * h * w, chunk_t, c)
        )
        query_hidden = self.register_in(query)
        kv_hidden = self.chunk_in(key_value)
        update, _ = self.cross_attn(
            self.query_norm(query_hidden),
            self.kv_norm(kv_hidden),
            self.kv_norm(kv_hidden),
            need_weights=False,
        )
        query = query + self.out(update)
        query = query + self.ffn(self.channel_norm(query))
        return (
            query.reshape(b, h, w, register_t, c)
            .permute(0, 4, 3, 1, 2)
            .contiguous()
        )


class SpatialRegisterMemory(nn.Module):
    """固定 `[B,16,4,H,W]` 的 Register；它不是 VAE 视频 latent。"""

    def __init__(
        self,
        channels: int = 16,
        register_frames: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_update_layers: int = 2,
        chunk_time_tokens: int = 8,
    ) -> None:
        super().__init__()
        if register_frames != 4:
            raise ValueError("A 方案的 register_frames 固定为 4")
        self.channels = channels
        self.register_frames = register_frames
        self.chunk_time_tokens = chunk_time_tokens
        # Extractor 与 Updater 参数分离。Register 初值完全来自第一个
        # History Chunk，不保存 learnable scene state。
        self.extract_blocks = nn.ModuleList(
            [
                SpatialRegisterUpdateBlock(channels, hidden_dim, num_heads)
                for _ in range(num_update_layers)
            ]
        )
        self.update_blocks = nn.ModuleList(
            [
                SpatialRegisterUpdateBlock(channels, hidden_dim, num_heads)
                for _ in range(num_update_layers)
            ]
        )

    def extract(
        self,
        chunk_latent: torch.Tensor,
        *,
        detach_history: bool = False,
    ) -> torch.Tensor:
        """从首个 Chunk 内容直接建立 `[B,C,4,H,W]` Register。"""
        weight_dtype = self.extract_blocks[0].register_in.weight.dtype
        registers = F.adaptive_avg_pool3d(
            chunk_latent.float(),
            (self.register_frames, *chunk_latent.shape[-2:]),
        ).to(weight_dtype)
        chunk_tokens = F.adaptive_avg_pool3d(
            chunk_latent.float(),
            (self.chunk_time_tokens, *chunk_latent.shape[-2:]),
        ).to(weight_dtype)
        for block in self.extract_blocks:
            registers = block(registers, chunk_tokens)
        return registers.detach() if detach_history else registers

    def update(
        self,
        registers: torch.Tensor,
        chunk_latent: torch.Tensor,
        *,
        detach_history: bool = False,
    ) -> torch.Tensor:
        if registers.shape[-2:] != chunk_latent.shape[-2:]:
            raise ValueError("Register 与 chunk latent 的 H,W 必须一致")
        chunk_latent = F.adaptive_avg_pool3d(
            chunk_latent.float(),
            (self.chunk_time_tokens, *chunk_latent.shape[-2:]),
        ).to(registers.dtype)
        for block in self.update_blocks:
            registers = block(registers, chunk_latent)
        return registers.detach() if detach_history else registers
