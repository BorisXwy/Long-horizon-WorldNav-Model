"""GigaPolicy-style block-causal self-attention for the shared Wan blocks.

This module patches only the ``forward`` method of the existing Wan
self-attention modules.  It does not add parameters and therefore keeps the
official Wan checkpoint/state-dict layout unchanged.
"""

from __future__ import annotations

from types import MethodType

import torch


def _giga_causal_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    token_ignore_mask=None,
    dtype=torch.bfloat16,
):
    """Apply the causal relation ``context -> action -> future``.

    The surrounding Wan block invokes self-attention twice: once on the clean
    prefix plus action slots, and once for noisy-future queries over the full
    sequence.  We preserve that execution pattern while making the K/V sets
    explicit:

    - clean visual queries read clean visual tokens only;
    - state/action queries read clean visual plus state/action tokens;
    - noisy-future queries read every clean visual, state/action and future
      token.
    """

    del dtype
    dit_module = __import__("infworld.models.dit_model", fromlist=["flash_attention"])
    flash_attention = dit_module.flash_attention
    b, s = x.shape[:2]
    n, d = self.num_heads, self.head_dim
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)
    q, k = self._apply_rope_with_prefix_action_tokens(q, k, grid_sizes, freqs)

    if token_ignore_mask is not None:
        select = (~token_ignore_mask).unsqueeze(-1).unsqueeze(-1).to(k.dtype)
        k = k * select
        v = v * select

    if self.enable_context_parallel:
        raise NotImplementedError("GigaNav block-causal attention does not support context parallelism")

    num_c = int(getattr(self, "num_c", 0) or 0)
    num_video_c = int(getattr(self, "num_video_c", num_c) or 0)
    num_extra = int(getattr(self, "num_extra_prefix", 0) or 0)

    if num_extra > 0 and s == num_c:
        parts = []
        if num_video_c > 0:
            parts.append(
                flash_attention(
                    q[:, :num_video_c],
                    k[:, :num_video_c],
                    v[:, :num_video_c],
                    window_size=self.window_size,
                )
            )
        if num_c > num_video_c:
            parts.append(
                flash_attention(
                    q[:, num_video_c:num_c],
                    k[:, :num_c],
                    v[:, :num_c],
                    window_size=self.window_size,
                )
            )
        attended = torch.cat(parts, dim=1).type_as(x)
    elif num_c > 0 and num_c < s:
        attended = flash_attention(
            q[:, num_c:],
            k,
            v,
            window_size=self.window_size,
        ).type_as(x)
    else:
        attended = flash_attention(
            q,
            k,
            v,
            k_lens=seq_lens,
            window_size=self.window_size,
        ).type_as(x)
    return self.o(attended.flatten(2))


def install_giga_causal_attention(backbone: torch.nn.Module) -> None:
    """Install the parameter-free Giga causal relation on every Wan block."""

    for block in backbone.blocks:
        block.self_attn.forward = MethodType(_giga_causal_forward, block.self_attn)
        block.self_attn.giga_causal_attention = True
