"""Formal V1 WorldNav model used by full-pipeline smoke tests.

This module implements the current V1 architecture contract without using the
old toy model path:

- fixed ``R_null`` initial register template;
- unified ``RegisterCell`` for first write and recurrent updates;
- ``A_hist`` participates as independent tokens / cross-attention context;
- no action/latent/additive bias path;
- dual-stream visual/action backbone;
- video generation head, action flow decoder and 3D probe.

The implementation is intentionally configurable, so smoke tests can use small
hidden sizes while still executing the same model/data-flow semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .heads import FutureLatentHead
from .masks import V1TokenLayout, assert_policy_safe, build_policy_safe_attention_mask
from .stems import VisualPatchStem


@dataclass(slots=True)
class V1FullModelConfig:
    latent_channels: int = 16
    hidden_dim: int = 128
    action_dim: int = 6
    action_horizon: int = 10
    num_primitives: int = 12
    num_register_tokens: int = 16
    num_register_layers: int = 2
    num_backbone_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    patch_size: tuple[int, int, int] = (1, 2, 2)
    max_action_horizon: int = 64
    max_text_tokens: int = 64
    geometry_dim: int = 7

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["patch_size"] = list(self.patch_size)
        return payload


class MLP(nn.Module):
    def __init__(self, dim: int, *, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegisterCellBlock(nn.Module):
    """One no-bias recurrent register update block."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.register_self_norm = nn.LayerNorm(hidden_dim)
        self.register_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, ratio=mlp_ratio)

    def forward(self, register: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = self.register_self_norm(register)
        delta, _ = self.register_self_attn(query, query, query, need_weights=False)
        register = register + delta
        delta, _ = self.context_attn(
            self.context_norm(register),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        register = register + delta
        return register + self.ffn(self.ffn_norm(register))


class RegisterCell(nn.Module):
    """Unified register cell with fixed ``R_null`` template."""

    def __init__(self, cfg: V1FullModelConfig) -> None:
        super().__init__()
        self.num_register_tokens = cfg.num_register_tokens
        self.hidden_dim = cfg.hidden_dim
        self.blocks = nn.ModuleList(
            [
                RegisterCellBlock(cfg.hidden_dim, cfg.num_heads, cfg.mlp_ratio)
                for _ in range(cfg.num_register_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.register_type = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)

        slot = torch.arange(cfg.num_register_tokens, dtype=torch.float32)[:, None]
        freq = torch.arange(cfg.hidden_dim, dtype=torch.float32)[None]
        template = torch.sin(slot / (10000 ** (2 * (freq // 2) / max(cfg.hidden_dim, 1))))
        template[:, 1::2] = torch.cos(
            slot / (10000 ** (2 * (freq[:, 1::2] // 2) / max(cfg.hidden_dim, 1)))
        )
        self.register_buffer("r_null", template[None], persistent=False)

    def initial(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.r_null.to(device=device, dtype=dtype).expand(batch_size, -1, -1).contiguous()

    def forward(self, previous_register: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        register = previous_register + self.register_type.to(previous_register.dtype)
        for block in self.blocks:
            register = block(register, context_tokens)
        return self.out_norm(register)


class ActionTokenEncoder(nn.Module):
    def __init__(self, cfg: V1FullModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.primitive = nn.Embedding(cfg.num_primitives, cfg.hidden_dim)
        self.continuous = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.position = nn.Embedding(cfg.max_action_horizon, cfg.hidden_dim)
        self.kind = nn.Embedding(3, cfg.hidden_dim)  # hist / current / noise
        self.timestep = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)

    def _pos(self, horizon: int, *, device: torch.device) -> torch.Tensor:
        if horizon > self.cfg.max_action_horizon:
            raise ValueError(f"horizon {horizon} > max_action_horizon {self.cfg.max_action_horizon}")
        return self.position(torch.arange(horizon, device=device))[None]

    def from_primitives(
        self,
        primitive_ids: torch.Tensor,
        *,
        kind: Literal["hist", "current"] = "hist",
    ) -> torch.Tensor:
        kind_id = 0 if kind == "hist" else 1
        primitive_ids = primitive_ids.clamp_min(0).clamp_max(self.cfg.num_primitives - 1)
        tokens = self.primitive(primitive_ids.long())
        tokens = tokens + self._pos(tokens.shape[1], device=tokens.device).to(tokens.dtype)
        tokens = tokens + self.kind.weight[kind_id][None, None].to(tokens.dtype)
        return self.out_norm(tokens)

    def from_noise(self, action_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        tokens = self.continuous(action_noise)
        tokens = tokens + self._pos(tokens.shape[1], device=tokens.device).to(tokens.dtype)
        tokens = tokens + self.kind.weight[2][None, None].to(tokens.dtype)
        t = timestep.to(device=tokens.device, dtype=tokens.dtype).view(tokens.shape[0], 1, 1)
        tokens = tokens + self.timestep(t)
        return self.out_norm(tokens)


class DualStreamBlock(nn.Module):
    """Policy-safe dual-stream block.

    Visual stream contains ``[R_t, Z_obs, Z_future_noisy]``.  Action stream
    contains ``A_noise``.  Action tokens can read only safe visual prefix
    (register/obs/text); future visual tokens can read action/current-action
    condition.  This preserves the causal direction required by the project
    docs.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.visual_self = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.visual_ffn_norm = nn.LayerNorm(hidden_dim)
        self.visual_ffn = MLP(hidden_dim, ratio=mlp_ratio)

        self.action_norm = nn.LayerNorm(hidden_dim)
        self.action_self = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.action_to_context_norm = nn.LayerNorm(hidden_dim)
        self.action_context = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.action_ffn_norm = nn.LayerNorm(hidden_dim)
        self.action_ffn = MLP(hidden_dim, ratio=mlp_ratio)

        self.future_norm = nn.LayerNorm(hidden_dim)
        self.future_context = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(
        self,
        *,
        visual_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        layout: V1TokenLayout,
        safe_context_tokens: torch.Tensor,
        future_condition_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        allowed = build_policy_safe_attention_mask(layout, device=visual_tokens.device)
        assert_policy_safe(allowed, layout)
        attn_mask = ~allowed
        q = self.visual_norm(visual_tokens)
        delta, _ = self.visual_self(q, q, q, attn_mask=attn_mask, need_weights=False)
        visual_tokens = visual_tokens + delta
        visual_tokens = visual_tokens + self.visual_ffn(self.visual_ffn_norm(visual_tokens))

        q_action = self.action_norm(action_tokens)
        delta, _ = self.action_self(q_action, q_action, q_action, need_weights=False)
        action_tokens = action_tokens + delta
        if safe_context_tokens.shape[1] > 0:
            delta, _ = self.action_context(
                self.action_to_context_norm(action_tokens),
                safe_context_tokens,
                safe_context_tokens,
                need_weights=False,
            )
            action_tokens = action_tokens + delta
        action_tokens = action_tokens + self.action_ffn(self.action_ffn_norm(action_tokens))

        if layout.n_future > 0 and future_condition_tokens.shape[1] > 0:
            future = visual_tokens[:, layout.future]
            delta, _ = self.future_context(
                self.future_norm(future),
                future_condition_tokens,
                future_condition_tokens,
                need_weights=False,
            )
            visual_tokens = visual_tokens.clone()
            visual_tokens[:, layout.future] = future + delta
        return visual_tokens, action_tokens


class V1DualStreamBackbone(nn.Module):
    def __init__(self, cfg: V1FullModelConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [DualStreamBlock(cfg.hidden_dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.num_backbone_layers)]
        )
        self.visual_norm = nn.LayerNorm(cfg.hidden_dim)
        self.action_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        *,
        visual_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        layout: V1TokenLayout,
        safe_context_tokens: torch.Tensor,
        future_condition_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            visual_tokens, action_tokens = block(
                visual_tokens=visual_tokens,
                action_tokens=action_tokens,
                layout=layout,
                safe_context_tokens=safe_context_tokens,
                future_condition_tokens=future_condition_tokens,
            )
        return self.visual_norm(visual_tokens), self.action_norm(action_tokens)


class ActionFlowDecoder(nn.Module):
    def __init__(self, cfg: V1FullModelConfig) -> None:
        super().__init__()
        self.flow = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        self.primitive = nn.Linear(cfg.action_dim, cfg.num_primitives)

    def forward(self, action_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        velocity = self.flow(action_tokens)
        return {
            "action_velocity": velocity,
            "primitive_logits": self.primitive(velocity),
        }


class GeometryProbe(nn.Module):
    def __init__(self, cfg: V1FullModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.geometry_dim),
        )

    def forward(self, register_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(register_tokens.mean(dim=1))


class V1FullWorldNavModel(nn.Module):
    def __init__(self, cfg: V1FullModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or V1FullModelConfig()
        cfg = self.cfg
        self.visual_stem = VisualPatchStem(
            in_channels=cfg.latent_channels,
            hidden_dim=cfg.hidden_dim,
            patch_size=cfg.patch_size,
        )
        self.action_encoder = ActionTokenEncoder(cfg)
        self.register_cell = RegisterCell(cfg)
        self.visual_timestep = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.text_type = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.visual_type = nn.Embedding(3, cfg.hidden_dim)  # register / obs / future
        self.backbone = V1DualStreamBackbone(cfg)
        self.future_head = FutureLatentHead(
            hidden_dim=cfg.hidden_dim,
            out_channels=cfg.latent_channels,
            patch_size=cfg.patch_size,
        )
        self.action_decoder = ActionFlowDecoder(cfg)
        self.geometry_probe = GeometryProbe(cfg)

    def encode_visual(self, latent: torch.Tensor, *, visual_type: int) -> torch.Tensor:
        tokens = self.visual_stem(latent)
        tokens = tokens + self.visual_type.weight[visual_type][None, None].to(tokens.dtype)
        return tokens

    def roll_register(
        self,
        history_latents: torch.Tensor,
        a_hist_primitives: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Roll register over history chunks.

        Args:
            history_latents: ``[B,S,C,T,H,W]``.
            a_hist_primitives: ``[B,S,H_action]``.
        """

        batch, steps = history_latents.shape[:2]
        dtype = next(self.parameters()).dtype
        register = self.register_cell.initial(batch, device=history_latents.device, dtype=dtype)
        states: list[torch.Tensor] = []
        for index in range(steps):
            visual = self.encode_visual(history_latents[:, index].to(dtype), visual_type=1)
            a_hist = self.action_encoder.from_primitives(a_hist_primitives[:, index], kind="hist")
            context = torch.cat([visual, a_hist], dim=1)
            register = self.register_cell(register, context)
            states.append(register)
        return register, states

    def _empty_text(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(batch, 0, self.cfg.hidden_dim, device=device, dtype=dtype)

    def _prepare_text(self, text_tokens: torch.Tensor | None, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if text_tokens is None:
            return self._empty_text(batch, device=device, dtype=dtype)
        return text_tokens.to(device=device, dtype=dtype) + self.text_type.to(dtype)

    def forward_core(
        self,
        *,
        register: torch.Tensor,
        z_obs: torch.Tensor,
        a_noise: torch.Tensor,
        action_timestep: torch.Tensor,
        z_future_noisy: torch.Tensor | None = None,
        visual_timestep: torch.Tensor | None = None,
        a_cur_primitives: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | V1TokenLayout | None]:
        dtype = next(self.parameters()).dtype
        batch = z_obs.shape[0]
        obs_tokens = self.encode_visual(z_obs.to(dtype), visual_type=1)
        future_tokens = None
        if z_future_noisy is not None:
            future_tokens = self.encode_visual(z_future_noisy.to(dtype), visual_type=2)
            if visual_timestep is not None:
                t = visual_timestep.to(device=z_obs.device, dtype=dtype).view(batch, 1, 1)
                future_tokens = future_tokens + self.visual_timestep(t)

        register_tokens = register.to(dtype) + self.visual_type.weight[0][None, None].to(dtype)
        visual_pieces = [register_tokens, obs_tokens]
        if future_tokens is not None:
            visual_pieces.append(future_tokens)
        visual_tokens = torch.cat(visual_pieces, dim=1)

        action_tokens = self.action_encoder.from_noise(a_noise.to(dtype), action_timestep)
        text = self._prepare_text(text_tokens, batch, z_obs.device, dtype)
        safe_context = torch.cat([register_tokens, obs_tokens, text], dim=1)

        future_condition_pieces = [action_tokens, text]
        if a_cur_primitives is not None:
            future_condition_pieces.append(self.action_encoder.from_primitives(a_cur_primitives, kind="current"))
        future_condition = torch.cat(future_condition_pieces, dim=1)

        layout = V1TokenLayout(
            n_register=register_tokens.shape[1],
            n_obs=obs_tokens.shape[1],
            n_future=0 if future_tokens is None else future_tokens.shape[1],
            n_action=0,
        )
        visual_hidden, action_hidden = self.backbone(
            visual_tokens=visual_tokens,
            action_tokens=action_tokens,
            layout=layout,
            safe_context_tokens=safe_context,
            future_condition_tokens=future_condition,
        )
        action_out = self.action_decoder(action_hidden)
        future_velocity = None
        if z_future_noisy is not None:
            future_hidden = visual_hidden[:, layout.future]
            future_velocity = self.future_head(
                future_hidden,
                latent_shape=(z_future_noisy.shape[2], z_future_noisy.shape[3], z_future_noisy.shape[4]),
            )
        return {
            "visual_hidden": visual_hidden,
            "action_hidden": action_hidden,
            "future_velocity": future_velocity,
            "action_velocity": action_out["action_velocity"],
            "primitive_logits": action_out["primitive_logits"],
            "register_after_backbone": visual_hidden[:, layout.register],
            "layout": layout,
        }

    def forward_stage1(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        register, _ = self.roll_register(batch["history_latents"], batch["a_hist_primitives"])
        out = self.forward_core(
            register=register,
            z_obs=batch["z_obs"],
            z_future_noisy=batch["z_future_noisy"],
            visual_timestep=batch["visual_timestep"],
            a_cur_primitives=batch["a_cur_primitives"],
            a_noise=batch["a_noise"],
            action_timestep=batch["action_timestep"],
            text_tokens=batch.get("text_tokens"),
        )
        visual_target = batch["z_future_target"].to(out["future_velocity"].dtype) - batch["z_future_noisy"].to(
            out["future_velocity"].dtype
        )
        visual_loss = F.mse_loss(out["future_velocity"], visual_target)
        return {
            **out,
            "loss": visual_loss,
            "loss_visual": visual_loss.detach(),
            "loss_action": torch.zeros((), device=visual_loss.device, dtype=visual_loss.dtype),
        }

    def forward_stage2(self, batch: dict[str, torch.Tensor], *, lambda_3d: float = 0.1) -> dict[str, torch.Tensor]:
        out = self.forward_stage1(batch)
        pred_geometry = self.geometry_probe(out["register_after_backbone"])
        geometry_loss = F.mse_loss(pred_geometry, batch["geometry_target"].to(pred_geometry.dtype))
        loss = out["loss"] + float(lambda_3d) * geometry_loss
        return {
            **out,
            "geometry_pred": pred_geometry,
            "loss": loss,
            "loss_3d": geometry_loss.detach(),
        }

    def forward_stage3(
        self,
        batch: dict[str, torch.Tensor],
        *,
        lambda_ce: float = 0.1,
        lambda_video: float = 0.25,
        lambda_3d: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Stage Three policy loss with Stage One/Two rehearsal losses.

        The same dual-stream backbone call is used when video fields are present:
        action tokens still cannot attend to future noisy visual tokens, while
        the video generation branch can read ``A_cur`` / ``A_noise`` as
        conditions.  This matches the final plan: Stage Three learns policy
        outputs and keeps videogen / 3D representations alive.
        """

        register, _ = self.roll_register(batch["history_latents"], batch["a_hist_primitives"])
        has_video = "z_future_noisy" in batch and "z_future_target" in batch
        out = self.forward_core(
            register=register,
            z_obs=batch["z_obs"],
            z_future_noisy=batch.get("z_future_noisy") if has_video else None,
            visual_timestep=batch.get("visual_timestep") if has_video else None,
            a_cur_primitives=batch.get("a_cur_primitives") if has_video else None,
            a_noise=batch["a_noise"],
            action_timestep=batch["action_timestep"],
            text_tokens=batch.get("text_tokens"),
        )
        target_velocity = batch["action_target"].to(out["action_velocity"].dtype) - batch["a_noise"].to(
            out["action_velocity"].dtype
        )
        action_flow_loss = F.mse_loss(out["action_velocity"], target_velocity)
        ce_loss = F.cross_entropy(
            out["primitive_logits"].reshape(-1, self.cfg.num_primitives),
            batch["action_primitives"].reshape(-1).long(),
        )
        loss = action_flow_loss + float(lambda_ce) * ce_loss
        extra_terms: dict[str, torch.Tensor] = {}
        if has_video:
            visual_target = batch["z_future_target"].to(out["future_velocity"].dtype) - batch["z_future_noisy"].to(
                out["future_velocity"].dtype
            )
            visual_loss = F.mse_loss(out["future_velocity"], visual_target)
            loss = loss + float(lambda_video) * visual_loss
            extra_terms["loss_visual_rehearsal"] = visual_loss.detach()
        if "geometry_target" in batch:
            pred_geometry = self.geometry_probe(out["register_after_backbone"])
            geometry_loss = F.mse_loss(pred_geometry, batch["geometry_target"].to(pred_geometry.dtype))
            loss = loss + float(lambda_3d) * geometry_loss
            extra_terms["geometry_pred"] = pred_geometry
            extra_terms["loss_3d_rehearsal"] = geometry_loss.detach()
        return {
            **out,
            "loss": loss,
            "loss_action_flow": action_flow_loss.detach(),
            "loss_ce_aux": ce_loss.detach(),
            **extra_terms,
        }

    @torch.no_grad()
    def infer_videogen(self, batch: dict[str, torch.Tensor], *, step_size: float = 1.0) -> dict[str, torch.Tensor]:
        register, _ = self.roll_register(batch["history_latents"], batch["a_hist_primitives"])
        out = self.forward_core(
            register=register,
            z_obs=batch["z_obs"],
            z_future_noisy=batch["z_future_noisy"],
            visual_timestep=batch["visual_timestep"],
            a_cur_primitives=batch["a_cur_primitives"],
            a_noise=batch["a_noise"],
            action_timestep=batch["action_timestep"],
            text_tokens=batch.get("text_tokens"),
        )
        z_future = batch["z_future_noisy"].to(out["future_velocity"].dtype) + float(step_size) * out["future_velocity"]
        return {"z_future": z_future, "future_velocity": out["future_velocity"]}

    @torch.no_grad()
    def infer_policy(self, batch: dict[str, torch.Tensor], *, step_size: float = 1.0) -> dict[str, torch.Tensor]:
        register, _ = self.roll_register(batch["history_latents"], batch["a_hist_primitives"])
        out = self.forward_core(
            register=register,
            z_obs=batch["z_obs"],
            a_noise=batch["a_noise"],
            action_timestep=batch["action_timestep"],
            text_tokens=batch.get("text_tokens"),
        )
        action_chunk = batch["a_noise"].to(out["action_velocity"].dtype) + float(step_size) * out["action_velocity"]
        return {
            "action_chunk": action_chunk,
            "primitive_logits": out["primitive_logits"],
            "primitive_ids": out["primitive_logits"].argmax(dim=-1),
        }

    def structural_audit(self) -> dict[str, object]:
        names = [name for name, _ in self.named_modules()]
        bad_keywords = ["history_to_latent", "bias_injector", "latent_bias", "action_bias"]
        return {
            "uses_register_cell": any(isinstance(m, RegisterCell) for m in self.modules()),
            "has_register_extractor_module": any("extractor" in name.lower() for name in names),
            "has_register_updater_module": any("updater" in name.lower() for name in names),
            "has_additive_action_bias_path": any(any(key in name.lower() for key in bad_keywords) for name in names),
            "config": self.cfg.to_dict(),
        }
