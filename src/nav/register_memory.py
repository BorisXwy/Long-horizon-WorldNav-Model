"""流式 Register memory。

每个视频 chunk 编码为少量观测 token；持久化 Register 以 query 身份通过
cross-attention 读取这些 token。模块提供两种注入形式：

1. latent_prefix：投影成 VAE latent plane，沿时间轴放在噪声 latent 前；
2. dit_condition：投影成 UMT5 同维 token，追加到 DiT cross-attention 条件。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RegisterUpdateBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.register_norm = nn.LayerNorm(dim)
        self.chunk_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=0.0
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(
        self, registers: torch.Tensor, chunk_tokens: torch.Tensor
    ) -> torch.Tensor:
        update, _ = self.cross_attn(
            self.register_norm(registers),
            self.chunk_norm(chunk_tokens),
            self.chunk_norm(chunk_tokens),
            need_weights=False,
        )
        registers = registers + update
        return registers + self.ffn(self.ffn_norm(registers))


class RegisterMemory(nn.Module):
    """有界、可跨 chunk 递归传递的 Register 状态。"""

    def __init__(
        self,
        latent_channels: int = 16,
        register_dim: int = 256,
        num_registers: int = 16,
        num_update_layers: int = 2,
        num_heads: int = 8,
        caption_channels: int = 4096,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.register_dim = register_dim
        self.num_registers = num_registers

        self.chunk_projection = nn.Sequential(
            nn.Linear(latent_channels, register_dim),
            nn.LayerNorm(register_dim),
        )
        self.extract_blocks = nn.ModuleList(
            [
                RegisterUpdateBlock(register_dim, num_heads)
                for _ in range(num_update_layers)
            ]
        )
        self.update_blocks = nn.ModuleList(
            [
                RegisterUpdateBlock(register_dim, num_heads)
                for _ in range(num_update_layers)
            ]
        )
        self.latent_projection = nn.Linear(register_dim, latent_channels)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(register_dim),
            nn.Linear(register_dim, caption_channels),
        )

    def extract(
        self,
        chunk_latent: torch.Tensor,
        *,
        detach_history: bool = False,
    ) -> torch.Tensor:
        """从首个 Chunk 内容直接提取16个 Register，不使用可学习初值。"""
        # 4×2×2 = 16，确保各 Register 来自不同的时空观测单元。
        pooled = F.adaptive_avg_pool3d(chunk_latent.float(), (4, 2, 2))
        registers = pooled.flatten(2).transpose(1, 2).to(
            self.chunk_projection[0].weight.dtype
        )
        registers = self.chunk_projection(registers)
        chunk_tokens = self.tokenize_chunk(chunk_latent).to(registers.dtype)
        for block in self.extract_blocks:
            registers = block(registers, chunk_tokens)
        return registers.detach() if detach_history else registers

    def tokenize_chunk(
        self, chunk_latent: torch.Tensor, token_grid: tuple[int, int, int] = (4, 4, 4)
    ) -> torch.Tensor:
        if chunk_latent.ndim != 5:
            raise ValueError("chunk_latent 必须是 [B,C,T,H,W]")
        pooled = F.adaptive_avg_pool3d(chunk_latent.float(), token_grid)
        tokens = pooled.flatten(2).transpose(1, 2).to(
            self.chunk_projection[0].weight.dtype
        )
        return self.chunk_projection(tokens)

    def update(
        self,
        registers: torch.Tensor,
        chunk_latent: torch.Tensor,
        *,
        detach_history: bool = False,
    ) -> torch.Tensor:
        chunk_tokens = self.tokenize_chunk(chunk_latent).to(registers.dtype)
        for block in self.update_blocks:
            registers = block(registers, chunk_tokens)
        return registers.detach() if detach_history else registers

    def as_latent_prefix(
        self, registers: torch.Tensor, spatial_size: tuple[int, int]
    ) -> torch.Tensor:
        """返回 [B,C,R,H,W]；R 个 Register 对应 R 个 latent time planes。"""
        height, width = spatial_size
        prefix = self.latent_projection(registers).transpose(1, 2)
        return prefix[:, :, :, None, None].expand(-1, -1, -1, height, width)

    def as_dit_condition(self, registers: torch.Tensor) -> torch.Tensor:
        """返回 [B,R,caption_channels]，可追加到 UMT5 token 后。"""
        return self.condition_projection(registers)
