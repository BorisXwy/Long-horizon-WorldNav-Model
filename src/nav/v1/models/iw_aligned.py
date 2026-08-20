"""IW/Wan-aligned V1 model path.

这个模块用于把当前 V1 Stage2/Stage3 的训练重新对齐到旧版可收敛的
InfiniteWorld/Wan 原生 generation branch：

- DiT 主干、patch embedding、time embedding、cross-attention 和 denoise head
  全部使用 `infworld.models.dit_model.WanModel`；
- HPMC/latent_encoder 仍被移除，history 由有界 Register prefix 提供；
- `A_cur` 作为 Wan cross-attention condition，`A_query/A_noise slot` 作为
  shared action tokens 进入同一 Wan token stream；
- Stage2 pose probe 从 clean current visual prefix hidden 读取。

该路径的目的不是恢复旧 A/B 的全部语义，而是把“视频生成分支”先恢复到
Wan/IW 原生计算，再继续承载 V1 的 register/action/3D 接口。
"""

from __future__ import annotations

import sys
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

_THIS = Path(__file__).resolve()
_IW_ROOT = Path(os.environ.get("NAV_INF_WORLD_ROOT", str(_THIS.parents[5] / "Infinite-World-legacy20")))
if str(_IW_ROOT) not in sys.path:
    sys.path.insert(0, str(_IW_ROOT))

import infworld.models.dit_model as dit_model_module  # noqa: E402
from infworld.models.dit_model import WanModel  # noqa: E402

from .heads import FramePoseHead


def _sdpa_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
) -> torch.Tensor:
    """PyTorch SDPA fallback for environments without flash-attn.

    Wan passes tensors as `[B,L,H,D]`; SDPA expects `[B,H,L,D]`.  We preserve
    the same public signature as `dit_model.flash_attention`.
    """

    del window_size, deterministic, version
    out_dtype = q.dtype
    if q_scale is not None:
        q = q * q_scale
    bsz, lq_max = q.shape[:2]
    lk_max = k.shape[1]
    out = torch.zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[-1], device=q.device, dtype=v.dtype)
    if q_lens is None:
        q_lens = torch.full((bsz,), lq_max, device=q.device, dtype=torch.long)
    if k_lens is None:
        k_lens = torch.full((bsz,), lk_max, device=k.device, dtype=torch.long)
    for idx in range(bsz):
        lq = int(q_lens[idx].item())
        lk = int(k_lens[idx].item())
        qi = q[idx : idx + 1, :lq].to(dtype).transpose(1, 2)
        ki = k[idx : idx + 1, :lk].to(dtype).transpose(1, 2)
        vi = v[idx : idx + 1, :lk].to(dtype).transpose(1, 2)
        oi = F.scaled_dot_product_attention(
            qi,
            ki,
            vi,
            dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        out[idx : idx + 1, :lq] = oi.transpose(1, 2)
    return out.to(out_dtype)


def install_attention_fallback_if_needed() -> None:
    if not (dit_model_module.FLASH_ATTN_2_AVAILABLE or dit_model_module.FLASH_ATTN_3_AVAILABLE):
        dit_model_module.flash_attention = _sdpa_attention_fallback


install_attention_fallback_if_needed()


@dataclass(slots=True)
class IWAlignedConfig:
    latent_channels: int = 16
    hidden_dim: int = 1536
    action_dim: int = 6
    ffn_dim: int = 8960
    freq_dim: int = 256
    num_heads: int = 12
    num_layers: int = 30
    caption_channels: int = 4096
    model_max_length: int = 512
    register_frames: int = 4
    register_hidden_dim: int = 64
    register_heads: int = 4
    register_layers: int = 2
    chunk_time_tokens: int = 4
    action_horizon: int = 10
    action_vocab_size: int = 12
    combo_action_vocab_size: int = 144
    pose_head_variant: str = "mlp"
    include_current_obs_prefix: bool = False
    update_register_with_current: bool = False
    pose_probe_mode: str = "separate_current_clean_pass"
    register_mode: str = "legacy_extract_update"
    current_action_mode: str = "legacy_iw_move_view"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpatialRegisterUpdateBlock(nn.Module):
    """Spatial latent-like register update used by the IW-compatible prefix."""

    def __init__(self, channels: int, dim: int, heads: int) -> None:
        super().__init__()
        self.register_in = nn.Linear(channels, dim)
        self.chunk_in = nn.Linear(channels, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.out = nn.Linear(dim, channels)
        self.channel_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(
        self,
        registers: torch.Tensor,
        chunk_latent: torch.Tensor,
        action_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, register_t, h, w = registers.shape
        chunk_t = chunk_latent.shape[2]
        query = registers.permute(0, 3, 4, 2, 1).reshape(b * h * w, register_t, c)
        key_value = chunk_latent.permute(0, 3, 4, 2, 1).reshape(b * h * w, chunk_t, c)
        if action_context is not None:
            if action_context.ndim != 3 or action_context.shape[0] != b or action_context.shape[2] != c:
                raise ValueError(
                    "action_context must be [B,K,C] matching register channels; "
                    f"got {tuple(action_context.shape)}, expected B={b}, C={c}"
                )
            action_tokens = (
                action_context[:, None, None]
                .expand(b, h, w, action_context.shape[1], c)
                .reshape(b * h * w, action_context.shape[1], c)
            )
            key_value = torch.cat([key_value, action_tokens.to(key_value.dtype)], dim=1)
        update, _ = self.cross_attn(
            self.query_norm(self.register_in(query)),
            self.kv_norm(self.chunk_in(key_value)),
            self.kv_norm(self.chunk_in(key_value)),
            need_weights=False,
        )
        query = query + self.out(update)
        query = query + self.ffn(self.channel_norm(query))
        return query.reshape(b, h, w, register_t, c).permute(0, 4, 3, 1, 2).contiguous()


class SpatialRegisterMemory(nn.Module):
    """Fixed `[B,16,4,H,W]` register prefix for Wan native `image_cond`.

    Two modes are intentionally supported:

    - ``legacy_extract_update`` keeps the fast-converging historical behavior:
      the first history chunk is pooled into a register-like latent and then
      refined by ``extract_blocks``; later chunks use separate ``update_blocks``.
    - ``fixed_rnull_unified`` is the formal V1 Register rule grafted onto the
      legacy20 Wan generation branch: start from a non-learned ``R_null`` tensor
      and use the same recurrent blocks for the first write and all later
      updates.  This changes only the Register memory path; the Wan video pass
      remains ``image_cond=registers`` and ``local_memory=z_obs latest``.
    """

    def __init__(self, cfg: IWAlignedConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.register_mode not in {"legacy_extract_update", "fixed_rnull_unified"}:
            raise ValueError(
                "register_mode must be one of "
                "{'legacy_extract_update', 'fixed_rnull_unified'}, "
                f"got {cfg.register_mode!r}"
            )
        self.extract_blocks = nn.ModuleList(
            [
                SpatialRegisterUpdateBlock(
                    cfg.latent_channels,
                    cfg.register_hidden_dim,
                    cfg.register_heads,
                )
                for _ in range(cfg.register_layers)
            ]
        )
        self.update_blocks = nn.ModuleList(
            [
                SpatialRegisterUpdateBlock(
                    cfg.latent_channels,
                    cfg.register_hidden_dim,
                    cfg.register_heads,
                )
                for _ in range(cfg.register_layers)
            ]
        )
        self.unified_blocks = nn.ModuleList(
            [
                SpatialRegisterUpdateBlock(
                    cfg.latent_channels,
                    cfg.register_hidden_dim,
                    cfg.register_heads,
                )
                for _ in range(cfg.register_layers)
            ]
        )

    def _pool_chunk(self, chunk_latent: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(
            chunk_latent.float(),
            (self.cfg.chunk_time_tokens, *chunk_latent.shape[-2:]),
        )

    def _fixed_rnull(
        self,
        chunk_latent: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        b, c, _, h, w = chunk_latent.shape
        return torch.zeros(
            b,
            c,
            self.cfg.register_frames,
            h,
            w,
            device=chunk_latent.device,
            dtype=dtype,
        )

    def _apply_blocks(
        self,
        blocks: nn.ModuleList,
        registers: torch.Tensor,
        chunk_latent: torch.Tensor,
        action_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        chunk = self._pool_chunk(chunk_latent).to(registers.dtype)
        action_context = action_context.to(registers.dtype) if action_context is not None else None
        for block in blocks:
            registers = block(registers, chunk, action_context=action_context)
        return registers

    def extract(self, chunk_latent: torch.Tensor, action_context: torch.Tensor | None = None) -> torch.Tensor:
        dtype = self.extract_blocks[0].register_in.weight.dtype
        if self.cfg.register_mode == "fixed_rnull_unified":
            registers = self._fixed_rnull(chunk_latent, dtype=dtype)
            return self._apply_blocks(self.unified_blocks, registers, chunk_latent, action_context=action_context)
        registers = F.adaptive_avg_pool3d(
            chunk_latent.float(),
            (self.cfg.register_frames, *chunk_latent.shape[-2:]),
        ).to(dtype)
        return self._apply_blocks(self.extract_blocks, registers, chunk_latent, action_context=action_context)

    def update(
        self,
        registers: torch.Tensor,
        chunk_latent: torch.Tensor,
        action_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        blocks = self.unified_blocks if self.cfg.register_mode == "fixed_rnull_unified" else self.update_blocks
        return self._apply_blocks(blocks, registers, chunk_latent, action_context=action_context)


class IWActionInterface(nn.Module):
    """V1 action slots attached to the native Wan token/context interfaces."""

    def __init__(self, cfg: IWAlignedConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.current_move_embedding = nn.Embedding(cfg.action_vocab_size, h)
        self.current_view_embedding = nn.Embedding(cfg.action_vocab_size, h)
        self.current_combo_embedding = nn.Embedding(cfg.combo_action_vocab_size, h)
        self.current_action_type = nn.Parameter(torch.randn(1, h) * 0.02)
        self.hist_primitive_latent = nn.Embedding(cfg.action_vocab_size, cfg.latent_channels)
        self.hist_combo_latent = nn.Embedding(cfg.combo_action_vocab_size, cfg.latent_channels)
        self.hist_position_latent = nn.Embedding(cfg.action_horizon, cfg.latent_channels)
        self.hist_action_norm = nn.LayerNorm(cfg.latent_channels)
        self.action_query = nn.Parameter(torch.randn(cfg.action_horizon, h) * 0.02)
        self.action_query_type = nn.Parameter(torch.randn(1, h) * 0.02)
        self.action_noise_in = nn.Linear(cfg.action_dim, h)
        self.action_position = nn.Embedding(cfg.action_horizon, h)
        self.action_timestep = nn.Sequential(
            nn.Linear(1, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.action_out_norm = nn.LayerNorm(h)
        self.action_velocity_head = nn.Linear(h, cfg.action_dim)
        self.primitive_head = nn.Linear(h, cfg.action_vocab_size)
        self.combo_head = nn.Linear(h, cfg.combo_action_vocab_size)
        self.move_head = nn.Linear(h, cfg.action_vocab_size)
        self.view_head = nn.Linear(h, cfg.action_vocab_size)

    def combine_trans_rot(self, trans: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        if trans.ndim == 1:
            trans = trans[:, None]
        if rot.ndim == 1:
            rot = rot[:, None]
        trans = trans.long().clamp_min(0).clamp_max(self.cfg.action_vocab_size - 1)
        rot = rot.long().clamp_min(0).clamp_max(self.cfg.action_vocab_size - 1)
        return (trans * self.cfg.action_vocab_size + rot).clamp_max(self.cfg.combo_action_vocab_size - 1)

    def current_condition(
        self,
        combo: torch.Tensor | None = None,
        *,
        trans: torch.Tensor | None = None,
        rot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode current video-generation action condition.

        Formal V1 action is one discrete combo token per time step:
        ``combo_id = trans_id * ACTION_BASE + rot_id``.  Historical
        InfiniteWorld names (move/view) are only a compatibility decomposition
        for the native video action encoder.
        """

        if combo is None:
            if trans is None or rot is None:
                raise ValueError("current_condition requires combo or trans+rot")
            combo = self.combine_trans_rot(trans, rot)
        if combo.ndim == 1:
            combo = combo[:, None]
        combo = combo.long().clamp_min(0).clamp_max(self.cfg.combo_action_vocab_size - 1)
        token = (
            self.current_combo_embedding(combo).mean(dim=1)
            + self.current_action_type[0]
        )
        return token[:, None]

    def query_tokens(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tokens = self.action_query[None].expand(batch_size, -1, -1) + self.action_query_type[None]
        return tokens.to(device=device, dtype=dtype)

    def hist_context_tokens(
        self,
        combo_ids: torch.Tensor,
        view_ids: torch.Tensor | None = None,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode ``A_hist`` as latent-channel tokens for Register cross-attention.

        Formal V1 uses one combo discrete action.  If an old caller passes
        separate move/view ids, ``view_ids`` is used only to build that combo
        id; it does not create a second independent action stream.
        """

        if view_ids is not None:
            combo_ids = self.combine_trans_rot(combo_ids, view_ids)
        if combo_ids.ndim == 1:
            combo_ids = combo_ids[:, None]
        if combo_ids.ndim != 2:
            raise ValueError(f"combo_ids must be [B] or [B,H], got {tuple(combo_ids.shape)}")
        horizon = min(combo_ids.shape[1], self.cfg.action_horizon)
        combo_ids = combo_ids[:, :horizon].long().clamp_min(0).clamp_max(self.cfg.combo_action_vocab_size - 1)
        tokens = self.hist_combo_latent(combo_ids)
        pos = self.hist_position_latent(torch.arange(horizon, device=combo_ids.device))[None]
        tokens = self.hist_action_norm(tokens + pos.to(tokens.dtype))
        return tokens.to(dtype)

    def noise_tokens(
        self,
        action_noise: torch.Tensor,
        action_timestep: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode formal ``A_noise(action_t)`` tokens for the shared Wan tail."""

        if action_noise.ndim == 2:
            action_noise = action_noise[:, None]
        if action_noise.ndim != 3 or action_noise.shape[-1] != self.cfg.action_dim:
            raise ValueError(
                f"action_noise must be [B,{self.cfg.action_dim}] or [B,H,{self.cfg.action_dim}], got {tuple(action_noise.shape)}"
            )
        horizon = action_noise.shape[1]
        if horizon > self.cfg.action_horizon:
            raise ValueError(f"action horizon {horizon} > configured {self.cfg.action_horizon}")
        tokens = self.action_noise_in(action_noise.to(self.action_noise_in.weight.dtype))
        pos = self.action_position(torch.arange(horizon, device=action_noise.device))[None]
        t = action_timestep.to(device=action_noise.device, dtype=tokens.dtype).view(action_noise.shape[0], 1, 1)
        tokens = tokens + pos.to(tokens.dtype) + self.action_timestep(t)
        return tokens.to(dtype)

    def decode(self, hidden: torch.Tensor | None) -> dict[str, torch.Tensor] | None:
        if hidden is None:
            return None
        hidden = self.action_out_norm(hidden[:, : self.cfg.action_horizon].to(self.action_out_norm.weight.dtype))
        return {
            "action_hidden": hidden,
            "action_velocity": self.action_velocity_head(hidden),
            "primitive_logits": self.primitive_head(hidden),
            "combo_logits": self.combo_head(hidden),
            "move_logits": self.move_head(hidden),
            "view_logits": self.view_head(hidden),
        }


class IWAlignedWorldNavModel(nn.Module):
    """Native Wan generation branch with V1 register/action/pose interfaces."""

    def __init__(self, cfg: IWAlignedConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or IWAlignedConfig()
        cfg = self.cfg
        if cfg.current_action_mode not in {"legacy_iw_move_view", "shared_tail_token", "none"}:
            raise ValueError(
                "current_action_mode must be one of "
                "{'legacy_iw_move_view', 'shared_tail_token', 'none'}, "
                f"got {cfg.current_action_mode!r}"
            )
        self.backbone = WanModel(
            model_type="t2v",
            dim=cfg.hidden_dim,
            # legacy20 WanModel concatenates a 4-channel condition/noise mask
            # inside forward, so the patch stem must consume 16 latent + 4 mask.
            in_channels=cfg.latent_channels + 4,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            out_channels=cfg.latent_channels,
            caption_channels=cfg.caption_channels,
            model_max_length=cfg.model_max_length,
            use_convenc=True,
        )
        self.register_memory = SpatialRegisterMemory(cfg)
        self.action_interface = IWActionInterface(cfg)
        self.pose_head = FramePoseHead(
            hidden_dim=cfg.hidden_dim,
            patch_size=(1, 2, 2),
            variant=cfg.pose_head_variant,
        )
        self.hmpc_removed = False

    def remove_hmpc(self) -> None:
        self.backbone.latent_encoder = nn.Identity()
        self.backbone.use_convenc = False
        self.hmpc_removed = True

    def load_wan_or_iw_checkpoint(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if path.suffix == ".safetensors":
            raw = load_file(str(path), device="cpu")
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            raw = payload.get("state_dict", payload)
        target = self.backbone.state_dict()
        state: dict[str, torch.Tensor] = {}
        partial_loaded: list[dict[str, Any]] = []
        for k, v in raw.items():
            if k not in target:
                continue
            if tuple(v.shape) == tuple(target[k].shape):
                state[k] = v
                continue
            if (
                k == "patch_embedding.weight"
                and v.ndim == target[k].ndim == 5
                and v.shape[0] == target[k].shape[0]
                and v.shape[2:] == target[k].shape[2:]
                and v.shape[1] < target[k].shape[1]
            ):
                expanded = target[k].clone()
                expanded.zero_()
                expanded[:, : v.shape[1]].copy_(v)
                state[k] = expanded
                partial_loaded.append(
                    {
                        "key": k,
                        "source_shape": list(v.shape),
                        "target_shape": list(target[k].shape),
                        "rule": "copy pretrained latent channels; zero initialize mask channels",
                    }
                )
        mismatched = [
            k
            for k, v in raw.items()
            if k in target and tuple(v.shape) != tuple(target[k].shape)
            and k not in state
        ]
        result = self.backbone.load_state_dict(state, strict=False)
        self.remove_hmpc()
        return {
            "path": str(path),
            "loaded_keys": len(state),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "mismatched_keys": mismatched,
            "partial_loaded": partial_loaded,
        }

    def empty_text(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.zeros(batch, 1, self.cfg.model_max_length, self.cfg.caption_channels, device=device, dtype=dtype)
        y_mask = torch.zeros(batch, self.cfg.model_max_length, device=device, dtype=dtype)
        return y, y_mask

    def _hist_action_context(
        self,
        a_hist_move: torch.Tensor | None,
        index: int,
        *,
        dtype: torch.dtype,
        a_hist_view: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if a_hist_move is None:
            return None
        if a_hist_move.ndim not in {2, 3}:
            raise ValueError(f"a_hist_move must be [B,S] or [B,S,H], got {tuple(a_hist_move.shape)}")
        if index >= a_hist_move.shape[1]:
            return None
        move = a_hist_move[:, index]
        view = None
        if a_hist_view is not None:
            if a_hist_view.ndim not in {2, 3}:
                raise ValueError(f"a_hist_view must be [B,S] or [B,S,H], got {tuple(a_hist_view.shape)}")
            if index < a_hist_view.shape[1]:
                view = a_hist_view[:, index]
        return self.action_interface.hist_context_tokens(move, view, dtype=dtype)

    def roll_register(
        self,
        history_latents: torch.Tensor,
        a_hist_move: torch.Tensor | None = None,
        a_hist_view: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = self.backbone.patch_embedding.weight.dtype
        first = history_latents[:, 0]
        registers = self.register_memory.extract(
            first,
            action_context=self._hist_action_context(a_hist_move, 0, dtype=dtype, a_hist_view=a_hist_view),
        )
        for idx in range(1, history_latents.shape[1]):
            registers = self.register_memory.update(
                registers,
                history_latents[:, idx],
                action_context=self._hist_action_context(
                    a_hist_move,
                    idx,
                    dtype=registers.dtype,
                    a_hist_view=a_hist_view,
                ),
            )
        return registers

    def _t_for_wan(self, visual_timestep: torch.Tensor) -> torch.Tensor:
        # Formal V1 stores normalized [0,1] RFlow timesteps; Wan time embedding
        # follows IW and expects [0,1000].
        t = visual_timestep.float()
        if float(t.detach().max().cpu()) <= 1.5:
            t = t * 1000.0
        return t

    def _pad_video_action_for_iw(self, action: torch.Tensor, *, length: int = 81) -> torch.Tensor:
        """Pad H_action policy chunks to InfiniteWorld's 81-frame action input.

        NAV policy/action tokens use H=10, but the restored InfiniteWorld
        video-action encoder is a strided Conv1d stack calibrated around the
        original 81-frame action sequence.  Feeding H=10 directly changes the
        action embedding temporal length and breaks Wan's visual token layout.
        """

        action = action.long()
        if action.ndim == 1:
            action = action[:, None]
        if action.shape[1] == length:
            return action
        if action.shape[1] > length:
            return action[:, -length:]
        pad = torch.zeros(action.shape[0], length - action.shape[1], device=action.device, dtype=action.dtype)
        return torch.cat([pad, action], dim=1)

    def _combo_from_legacy_fields(
        self,
        primary: torch.Tensor | None,
        secondary: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if primary is None:
            return None
        if secondary is None:
            return primary
        return self.action_interface.combine_trans_rot(primary, secondary)

    def _split_combo_for_iw(self, combo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combo = combo.long().clamp_min(0).clamp_max(self.cfg.combo_action_vocab_size - 1)
        move = (combo // self.cfg.action_vocab_size).clamp_max(self.cfg.action_vocab_size - 1)
        view = (combo % self.cfg.action_vocab_size).clamp_max(self.cfg.action_vocab_size - 1)
        return move, view

    def forward_core(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_prefix_hidden: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        if not self.hmpc_removed:
            raise RuntimeError("call load_wan_or_iw_checkpoint() before training")
        dtype = self.backbone.patch_embedding.weight.dtype
        history = batch["history_latents"].to(dtype)
        z_obs = batch["z_obs"].to(dtype)
        z_future_noisy = batch["z_future_noisy"].to(dtype)
        # Formal V1 has one combo discrete action.  Video datasets may still
        # land old move/view fields; those are immediately converted into the
        # same combo id.  The sample has one action tensor whose *purpose* is
        # condition for video data or label for VLN data.
        a_hist_combo = batch.get("a_hist_combo")
        if a_hist_combo is None:
            a_hist_combo = self._combo_from_legacy_fields(
                batch.get("a_hist_move", batch.get("a_hist_primitives")),
                batch.get("a_hist_view"),
            )
        if a_hist_combo is not None:
            a_hist_combo = a_hist_combo.to(device=z_obs.device)
        registers = self.roll_register(history, a_hist_combo, None)
        if self.cfg.update_register_with_current:
            registers = self.register_memory.update(registers, z_obs)
        # Old IW-aligned video branch uses Register prefix + latest-frame local
        # memory.  The full current chunk can be enabled only for probe/debug
        # experiments because it changes Wan's visual token layout.
        if self.cfg.include_current_obs_prefix:
            image_cond = torch.cat([registers.to(dtype), z_obs], dim=2)
        else:
            image_cond = registers.to(dtype)
        local_memory = z_obs[:, :, -1:].contiguous()
        b = z_obs.shape[0]
        y = batch.get("y")
        y_mask = batch.get("y_mask")
        if y is None or y_mask is None:
            y, y_mask = self.empty_text(b, device=z_obs.device, dtype=dtype)
        else:
            y = y.to(device=z_obs.device, dtype=dtype)
            y_mask = y_mask.to(device=z_obs.device, dtype=dtype)
        a_cur_combo = batch.get("a_cur_combo")
        if a_cur_combo is None:
            a_cur_combo = self._combo_from_legacy_fields(
                batch.get("a_cur_move", batch.get("a_cur_primitives")),
                batch.get("a_cur_view"),
            )
        if a_cur_combo is None:
            a_cur_combo = torch.zeros(b, self.cfg.action_horizon, device=z_obs.device, dtype=torch.long)
        a_cur_combo = a_cur_combo.to(device=z_obs.device)
        a_cur_move, a_cur_view = self._split_combo_for_iw(a_cur_combo)
        if self.cfg.current_action_mode == "legacy_iw_move_view":
            video_a_cur_move = self._pad_video_action_for_iw(a_cur_move, length=81)
            video_a_cur_view = self._pad_video_action_for_iw(a_cur_view, length=81)
            disable_native_action_embedding = False
        else:
            video_a_cur_move = torch.zeros(b, 81, device=z_obs.device, dtype=torch.long)
            video_a_cur_view = torch.zeros(b, 81, device=z_obs.device, dtype=torch.long)
            disable_native_action_embedding = True
        if "a_noise" in batch:
            action_timestep = batch.get(
                "action_timestep",
                torch.zeros(b, device=z_obs.device, dtype=torch.float32),
            )
            action_tokens = self.action_interface.noise_tokens(
                batch["a_noise"].to(device=z_obs.device),
                action_timestep.to(device=z_obs.device),
                dtype=dtype,
            )
        else:
            action_tokens = self.action_interface.query_tokens(b, device=z_obs.device, dtype=dtype)
        if self.cfg.current_action_mode == "shared_tail_token":
            current_tokens = self.action_interface.current_condition(a_cur_combo).to(
                device=z_obs.device,
                dtype=dtype,
            )
            action_tokens = torch.cat([action_tokens, current_tokens], dim=1)
        out = self.backbone(
            x=z_future_noisy,
            t=self._t_for_wan(batch["visual_timestep"].to(z_obs.device)),
            y=y,
            y_mask=y_mask,
            image_cond=image_cond,
            memory_is_precomputed=True,
            local_memory=local_memory,
            # A_cur keeps the restored InfiniteWorld native video-action
            # condition path.  The shared action tail below is policy-safe:
            # its rows are forbidden from reading future visual tokens inside
            # the patched legacy20 Wan self-attention.
            move=video_a_cur_move,
            view=video_a_cur_view,
            disable_native_action_embedding=disable_native_action_embedding,
            shared_action_tokens=action_tokens,
            return_shared_action_tokens=True,
            return_prefix_video_hidden=return_prefix_hidden,
        )
        if isinstance(out, torch.Tensor):
            video = out
            shared_action_hidden = None
            prefix_hidden = None
        else:
            video = out["video"]
            shared_action_hidden = out.get("shared_action_hidden")
            prefix_hidden = out.get("prefix_video_hidden")
        # legacy20 WanModel reconstructs the full visual token grid
        # `[condition/register/local, noisy_future]`.  Stage losses supervise
        # only the denoised future chunk, matching the old scheduler +
        # ignore-mask semantics.
        if video.shape[2] != z_future_noisy.shape[2]:
            video = video[:, :, -z_future_noisy.shape[2] :].contiguous()
        return {
            "future_velocity": video,
            "shared_action_hidden": shared_action_hidden,
            "action_outputs": self.action_interface.decode(shared_action_hidden),
            "prefix_video_hidden": prefix_hidden,
            "registers": registers,
        }

    @staticmethod
    def video_flow_target(batch: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        return batch["z_future_noise"].to(dtype) - batch["z_future_target"].to(dtype)

    def current_obs_hidden(self, prefix_video_hidden: torch.Tensor, z_obs: torch.Tensor) -> torch.Tensor:
        if not self.cfg.include_current_obs_prefix:
            raise RuntimeError(
                "current_obs_hidden requires include_current_obs_prefix=True; "
                "old IW-aligned video branch only exposes register/local prefix hidden"
            )
        # prefix hidden layout is patchified [register T=4, z_obs T=4, local T=1].
        h_tokens = z_obs.shape[-2] // 2
        w_tokens = z_obs.shape[-1] // 2
        obs_tokens = z_obs.shape[2] * h_tokens * w_tokens
        local_tokens = h_tokens * w_tokens
        return prefix_video_hidden[:, -(obs_tokens + local_tokens) : -local_tokens]

    def first_current_clean_hidden(self, prefix_video_hidden: torch.Tensor, z_obs: torch.Tensor) -> torch.Tensor:
        """Read current-clean chunk hidden from a separate pose-probe pass.

        The fast-converging legacy20 video branch keeps
        ``image_cond=Register`` and ``local_memory=latest latent frame`` in the
        generation pass.  Stage2 geometry supervision still needs hidden states
        aligned to the clean current chunk, so the legacy bridge uses a second
        Wan pass with ``image_cond=z_obs`` and a zero dummy future.  This helper
        preserves the same token slicing for the IW-aligned formal path.
        """

        h_tokens = z_obs.shape[-2] // 2
        w_tokens = z_obs.shape[-1] // 2
        obs_tokens = z_obs.shape[2] * h_tokens * w_tokens
        return prefix_video_hidden[:, :obs_tokens]

    def pose_probe_current_hidden(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run a Stage2 pose-probe pass without changing video loss layout."""

        if self.cfg.pose_probe_mode != "separate_current_clean_pass":
            raise ValueError(f"unsupported pose_probe_mode: {self.cfg.pose_probe_mode}")
        dtype = self.backbone.patch_embedding.weight.dtype
        z_obs = batch["z_obs"].to(dtype)
        b = z_obs.shape[0]
        y = batch.get("y")
        y_mask = batch.get("y_mask")
        if y is None or y_mask is None:
            y, y_mask = self.empty_text(b, device=z_obs.device, dtype=dtype)
        else:
            y = y.to(device=z_obs.device, dtype=dtype)
            y_mask = y_mask.to(device=z_obs.device, dtype=dtype)
        dummy_future = torch.zeros_like(z_obs)
        noop = torch.zeros(b, 81, device=z_obs.device, dtype=torch.long)
        probe_out = self.backbone(
            x=dummy_future,
            t=torch.zeros(b, device=z_obs.device, dtype=torch.float32),
            y=y,
            y_mask=y_mask,
            image_cond=z_obs,
            memory_is_precomputed=True,
            local_memory=z_obs[:, :, -1:].contiguous(),
            move=noop,
            view=noop,
            return_prefix_video_hidden=True,
        )
        if not isinstance(probe_out, dict) or probe_out.get("prefix_video_hidden") is None:
            raise RuntimeError("Wan pose probe did not return prefix_video_hidden")
        return self.first_current_clean_hidden(probe_out["prefix_video_hidden"], z_obs)

    def forward_stage2(
        self,
        batch: dict[str, torch.Tensor],
        *,
        lambda_pose: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        out = self.forward_core(batch, return_prefix_hidden=True)
        future_velocity = out["future_velocity"]
        visual_target = self.video_flow_target(batch, future_velocity.dtype)
        visual_loss = F.mse_loss(future_velocity, visual_target)
        pose_pred = None
        if lambda_pose > 0:
            if self.cfg.include_current_obs_prefix:
                obs_hidden = self.current_obs_hidden(out["prefix_video_hidden"], batch["z_obs"])
            else:
                obs_hidden = self.pose_probe_current_hidden(batch)
            obs_hidden = obs_hidden.to(next(self.pose_head.parameters()).dtype)
            latent_shape = (batch["z_obs"].shape[2], batch["z_obs"].shape[3], batch["z_obs"].shape[4])
            pose_pred = self.pose_head(obs_hidden, latent_shape=latent_shape)
            pose_target = batch["pose_target"].to(device=pose_pred.device, dtype=pose_pred.dtype)
            pose_mask = batch.get("pose_mask")
            if pose_mask is not None:
                mask = pose_mask.to(device=pose_pred.device, dtype=pose_pred.dtype)
                while mask.ndim < pose_pred.ndim:
                    mask = mask.unsqueeze(-1)
                pose_loss = (((pose_pred - pose_target) ** 2) * mask).sum() / mask.expand_as(pose_pred).sum().clamp_min(1.0)
            else:
                pose_loss = F.mse_loss(pose_pred, pose_target)
        else:
            pose_loss = torch.zeros((), device=visual_loss.device, dtype=visual_loss.dtype)
        loss = visual_loss + float(lambda_pose) * pose_loss
        return {
            **out,
            "obs_hidden": obs_hidden if lambda_pose > 0 else None,
            "pose_pred": pose_pred,
            "loss": loss,
            "loss_visual_tensor": visual_loss,
            "loss_pose_tensor": pose_loss,
            "loss_visual": visual_loss.detach(),
            "loss_pose": pose_loss.detach(),
            "loss_3d": pose_loss.detach(),
        }

    def _masked_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        target = target.to(device=pred.device, dtype=pred.dtype)
        if mask is None:
            return F.mse_loss(pred, target)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        while mask.ndim < pred.ndim:
            mask = mask.unsqueeze(-1)
        denom = mask.expand_as(pred).sum().clamp_min(1.0)
        return (((pred - target) ** 2) * mask).sum() / denom

    def forward_stage3_policy(
        self,
        batch: dict[str, torch.Tensor],
        *,
        lambda_ce: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Stage3 VLN policy/action loss on the legacy20 shared Wan action tail.

        This is policy-only with respect to losses: a dummy future visual token
        is still passed through Wan because the restored legacy20 block updates
        rows after the hist/local prefix.  No video loss is computed here.
        """

        out = self.forward_core(batch, return_prefix_hidden=False)
        action_outputs = out["action_outputs"]
        if action_outputs is None:
            raise RuntimeError("forward_stage3_policy requires shared action hidden")
        pred_velocity = action_outputs["action_velocity"]
        if "action_combo" in batch:
            # Formal V1 policy supervision is categorical over the unified
            # trans-rot combo action.  The continuous velocity head is kept as
            # a compatibility/ablation output but is not the default Stage3
            # action label when combo supervision is available.
            loss_action_flow = torch.zeros((), device=pred_velocity.device, dtype=pred_velocity.dtype)
            combo_logits = action_outputs["combo_logits"]
            combo_target = batch["action_combo"].to(device=combo_logits.device).long()
            if combo_target.ndim == 1:
                combo_logits = combo_logits[:, 0]
            ce_all = F.cross_entropy(
                combo_logits.reshape(-1, combo_logits.shape[-1]),
                combo_target.reshape(-1),
                reduction="none",
            ).view_as(combo_target)
        else:
            action_target = batch["action_target"]
            if action_target.ndim == 2:
                pred_velocity = pred_velocity[:, 0]
            if pred_velocity.shape != action_target.shape:
                raise RuntimeError(
                    f"action velocity shape {tuple(pred_velocity.shape)} != target {tuple(action_target.shape)}"
                )
            target_velocity = action_target.to(pred_velocity.dtype) - batch["a_noise"].to(pred_velocity.dtype)
            loss_action_flow = self._masked_mse(pred_velocity, target_velocity, batch.get("action_loss_mask"))
            primitive_logits = action_outputs["primitive_logits"]
            primitive_target = batch["action_primitives"].to(device=primitive_logits.device).long()
            if primitive_target.ndim == 1:
                primitive_logits = primitive_logits[:, 0]
            ce_all = F.cross_entropy(
                primitive_logits.reshape(-1, primitive_logits.shape[-1]),
                primitive_target.reshape(-1),
                reduction="none",
            ).view_as(primitive_target)
        mask = batch.get("action_loss_mask")
        if mask is not None:
            mask = mask.to(device=ce_all.device, dtype=ce_all.dtype)
            loss_ce = (ce_all * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss_ce = ce_all.mean()
        loss = loss_action_flow + float(lambda_ce) * loss_ce
        return {
            **out,
            "loss": loss,
            "loss_action_flow": loss_action_flow.detach(),
            "loss_ce_aux": loss_ce.detach(),
            "action_velocity": pred_velocity,
            "primitive_logits": action_outputs["primitive_logits"],
            "combo_logits": action_outputs["combo_logits"],
        }
