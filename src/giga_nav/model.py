"""Complete GigaWorld-style navigation model using Wan2.1-1.3B.

The implementation uses the project's real ``infworld.models.dit_model.WanModel``
for every visual/text transformer block.  Giga's causal ordering is expressed
by Wan's existing clean-prefix/action-prefix versus noisy-future split:

    [state, reference visual, action slots, noisy future visual]

The returned action hidden states are the unchanged shared Wan block outputs;
only a navigation classifier is added after them.  There is no toy backbone or
side-channel visual encoder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .config import GigaNavConfig
from .causal_attention import install_giga_causal_attention


def _load_wan_model_class():
    project_root = Path(__file__).resolve().parents[3]
    default_root = project_root / "Infinite-World"
    root = Path(os.environ.get("NAV_INF_WORLD_ROOT", str(default_root)))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import infworld.models.dit_model as dit_module

    # Keep this ablation independent of the legacy20 compatibility shim.  The
    # normal Infinite-World WanModel has the shared action-token API used here;
    # in this environment flash-attn is absent, so install the same SDPA
    # fallback used by the canonical V1 implementation.
    if not (dit_module.FLASH_ATTN_2_AVAILABLE or dit_module.FLASH_ATTN_3_AVAILABLE):
        def sdpa_fallback(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0,
                          softmax_scale=None, q_scale=None, causal=False,
                          window_size=(-1, -1), deterministic=False,
                          dtype=torch.bfloat16, version=None):
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
                lq_i, lk_i = int(q_lens[idx].item()), int(k_lens[idx].item())
                qi = q[idx:idx + 1, :lq_i].to(dtype).transpose(1, 2)
                ki = k[idx:idx + 1, :lk_i].to(dtype).transpose(1, 2)
                vi = v[idx:idx + 1, :lk_i].to(dtype).transpose(1, 2)
                oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p if torch.is_grad_enabled() else 0.0, is_causal=causal, scale=softmax_scale)
                out[idx:idx + 1, :lq_i] = oi.transpose(1, 2)
            return out.to(out_dtype)
        dit_module.flash_attention = sdpa_fallback
    from infworld.models.dit_model import WanModel

    return WanModel, root


class _ActionProjector(nn.Module):
    """Giga's MLP action/state projector with a learned role/position code."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, max_tokens: int, role_id: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.position = nn.Embedding(max_tokens, output_dim)
        self.role = nn.Parameter(torch.zeros(1, 1, output_dim))
        nn.init.normal_(self.role, std=0.02)
        self.role_id = role_id

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"expected [B,N,D] projector input, got {tuple(values.shape)}")
        if values.shape[1] > self.position.num_embeddings:
            raise ValueError(f"token count {values.shape[1]} exceeds {self.position.num_embeddings}")
        positions = self.position(torch.arange(values.shape[1], device=values.device))[None]
        return self.net(values) + positions + self.role


class GigaNavModel(nn.Module):
    """Giga token layout + Wan2.1-1.3B shared backbone + nav head."""

    def __init__(self, cfg: GigaNavConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GigaNavConfig()
        WanModel, self.infworld_root = _load_wan_model_class()
        self.backbone = WanModel(
            model_type="t2v",
            patch_size=self.cfg.patch_size,
            model_max_length=self.cfg.text_length,
            in_channels=self.cfg.latent_channels,
            dim=self.cfg.hidden_dim,
            ffn_dim=self.cfg.ffn_dim,
            freq_dim=self.cfg.freq_dim,
            caption_channels=self.cfg.caption_channels,
            out_channels=self.cfg.latent_channels,
            num_heads=self.cfg.num_heads,
            num_layers=self.cfg.num_layers,
            use_convenc=False,
        )
        dit_module = sys.modules["infworld.models.dit_model"]
        self.attention_backend = (
            "flash_attn"
            if (dit_module.FLASH_ATTN_2_AVAILABLE or dit_module.FLASH_ATTN_3_AVAILABLE)
            else "sdpa_fallback"
        )
        self.state_projector = _ActionProjector(
            self.cfg.state_input_dim,
            self.cfg.action_mlp_dim,
            self.cfg.hidden_dim,
            max_tokens=self.cfg.state_tokens,
            role_id=0,
        )
        self.action_projector = _ActionProjector(
            self.cfg.action_input_dim,
            self.cfg.action_mlp_dim,
            self.cfg.hidden_dim,
            max_tokens=self.cfg.action_horizon,
            role_id=1,
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.Linear(self.cfg.hidden_dim, self.cfg.num_nav_classes),
        )
        # Keep the small categorical readout numerically stable in fp32 even
        # when the 1.3B Wan backbone is moved to bf16.
        self.policy_head.float()
        if self.cfg.giga_causal_attention:
            install_giga_causal_attention(self.backbone)
        self.loaded_checkpoint: dict[str, Any] | None = None

    def load_wan_checkpoint(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load shape-compatible official Wan2.1 weights into the real backbone."""

        checkpoint = Path(path or self.cfg.backbone_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        raw = load_file(str(checkpoint), device="cpu") if checkpoint.suffix == ".safetensors" else torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        target = self.backbone.state_dict()
        compatible = {key: value for key, value in raw.items() if key in target and tuple(value.shape) == tuple(target[key].shape)}
        result = self.backbone.load_state_dict(compatible, strict=False)
        audit = {
            "path": str(checkpoint),
            "loaded_keys": len(compatible),
            "target_keys": len(target),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "shape_mismatch_keys": [key for key, value in raw.items() if key in target and tuple(value.shape) != tuple(target[key].shape)],
        }
        self.loaded_checkpoint = audit
        return audit

    def _prepare_text(self, y: torch.Tensor, y_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y.ndim == 3:
            y = y[:, None]
        if y.ndim != 4 or y.shape[2] != self.cfg.text_length or y.shape[3] != self.cfg.caption_channels:
            raise ValueError(
                "GigaNav expects UMT5 embeddings [B,1,512,4096], "
                f"got {tuple(y.shape)}"
            )
        # WanModel indexes ``y[:, 0]`` but expects ``y_mask`` to remain
        # [B,512], so do not add a singleton dimension to the mask.
        return y, y_mask

    def forward_policy(
        self,
        *,
        obs_latent: torch.Tensor,
        text_embedding: torch.Tensor,
        text_mask: torch.Tensor,
        state: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        visual_future: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run one complete Giga-style policy forward.

        ``obs_latent`` is a real T4 chunk ``[B,16,4,56,112]``.  Its first
        temporal latent is the clean reference; the remaining three temporal
        planes are the noisy-future slots.  R2R has no aligned robot state, so
        the adapter supplies a zero state token.  Action slots are zero-noise
        query slots for the discrete ablation, while preserving Giga's
        state/reference/action/future ordering and action-only readout.
        """

        if obs_latent.ndim != 5:
            raise ValueError(f"obs_latent must be [B,C,T,H,W], got {tuple(obs_latent.shape)}")
        b, c, t, h, w = obs_latent.shape
        expected = (self.cfg.latent_channels, self.cfg.latent_frames, self.cfg.latent_height, self.cfg.latent_width)
        if (c, t, h, w) != expected:
            raise ValueError(f"expected obs latent [B,{expected[0]},{expected[1]},{expected[2]},{expected[3]}], got {tuple(obs_latent.shape)}")
        y, y_mask = self._prepare_text(text_embedding, text_mask)
        dtype = self.backbone.patch_embedding.weight.dtype
        device = obs_latent.device
        obs_latent = obs_latent.to(device=device, dtype=dtype)
        ref_latent = obs_latent[:, :, :1]
        if visual_future is None:
            visual_future = torch.zeros_like(obs_latent[:, :, 1:]) if self.cfg.use_zero_future_visual else obs_latent[:, :, 1:]
        if visual_future.shape != obs_latent[:, :, 1:].shape:
            raise ValueError(f"visual_future shape {tuple(visual_future.shape)} != {tuple(obs_latent[:, :, 1:].shape)}")
        visual_future = visual_future.to(device=device, dtype=dtype)
        if state is None:
            state = torch.zeros(b, self.cfg.state_tokens, self.cfg.state_input_dim, device=device, dtype=dtype)
        if state.shape != (b, self.cfg.state_tokens, self.cfg.state_input_dim):
            raise ValueError(f"state must be {(b, self.cfg.state_tokens, self.cfg.state_input_dim)}, got {tuple(state.shape)}")
        if action_noise is None:
            action_noise = torch.zeros(b, self.cfg.action_horizon, self.cfg.action_input_dim, device=device, dtype=dtype)
        if action_noise.shape != (b, self.cfg.action_horizon, self.cfg.action_input_dim):
            raise ValueError(
                f"action_noise must be {(b, self.cfg.action_horizon, self.cfg.action_input_dim)}, got {tuple(action_noise.shape)}"
            )
        state_tokens = self.state_projector(state.to(dtype))
        action_tokens = self.action_projector(action_noise.to(dtype))
        shared_tokens = torch.cat([state_tokens, action_tokens], dim=1)
        zero_move = torch.zeros(b, 81, device=device, dtype=torch.long)
        out = self.backbone(
            x=visual_future,
            t=torch.zeros(b, device=device, dtype=torch.float32),
            y=y.to(device=device, dtype=dtype),
            y_mask=y_mask.to(device=device),
            image_cond=ref_latent,
            memory_is_precomputed=True,
            local_memory=ref_latent,
            move=zero_move,
            view=zero_move,
            disable_native_action_embedding=True,
            shared_action_tokens=shared_tokens,
            return_shared_action_tokens=True,
            policy_only=True,
            action_timestep=torch.zeros(b, device=device, dtype=torch.float32),
        )
        action_hidden_all = out.get("shared_action_hidden")
        if action_hidden_all is None or action_hidden_all.shape[1] < self.cfg.action_horizon:
            raise RuntimeError(f"Wan backbone returned invalid shared action hidden: {None if action_hidden_all is None else tuple(action_hidden_all.shape)}")
        action_hidden = action_hidden_all[:, -self.cfg.action_horizon :]
        self.policy_head.float()
        logits = self.policy_head(action_hidden.float())
        return {
            "action_logits": logits,
            "action_hidden": action_hidden,
            "all_prefix_action_hidden": action_hidden_all,
        }

    def action_classes_to_input(self, action_classes: torch.Tensor) -> torch.Tensor:
        """Encode discrete R2R actions in Giga's 14-D action input space."""

        if action_classes.ndim != 2 or action_classes.shape[1] != self.cfg.action_horizon:
            raise ValueError(
                f"action classes must be [B,{self.cfg.action_horizon}], got {tuple(action_classes.shape)}"
            )
        one_hot = F.one_hot(
            action_classes.long().clamp(0, self.cfg.num_nav_classes - 1),
            num_classes=self.cfg.num_nav_classes,
        )
        result = torch.zeros(
            *one_hot.shape[:-1],
            self.cfg.action_input_dim,
            device=one_hot.device,
            dtype=self.backbone.patch_embedding.weight.dtype,
        )
        result[..., : self.cfg.num_nav_classes] = one_hot.to(result.dtype)
        return result

    def forward_world_action(
        self,
        *,
        reference_latent: torch.Tensor,
        future_noisy: torch.Tensor,
        visual_timestep: torch.Tensor,
        text_embedding: torch.Tensor,
        text_mask: torch.Tensor,
        action_input: torch.Tensor,
        state: torch.Tensor | None = None,
        return_video: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run the shared AC-WM/WAM forward used by the YAML trainer.

        ``reference_latent`` is one clean causal-VAE plane and
        ``future_noisy`` contains two planes (eight RGB transitions).  Action
        slots are clean GT conditions for AC-WM and zero policy queries for
        WAM.  Both streams traverse the same Wan blocks.
        """

        if not self.cfg.giga_causal_attention:
            raise RuntimeError("video/cotrain requires giga_causal_attention=True")
        expected_ref = (
            self.cfg.latent_channels,
            1,
            self.cfg.latent_height,
            self.cfg.latent_width,
        )
        if reference_latent.ndim != 5 or tuple(reference_latent.shape[1:]) != expected_ref:
            raise ValueError(f"reference_latent must be [B,{','.join(map(str, expected_ref))}]")
        expected_future = (
            self.cfg.latent_channels,
            self.cfg.cotrain_future_latent_frames,
            self.cfg.latent_height,
            self.cfg.latent_width,
        )
        if future_noisy.ndim != 5 or tuple(future_noisy.shape[1:]) != expected_future:
            raise ValueError(f"future_noisy must be [B,{','.join(map(str, expected_future))}]")
        b = reference_latent.shape[0]
        if action_input.shape != (b, self.cfg.action_horizon, self.cfg.action_input_dim):
            raise ValueError(
                f"action_input must be {(b, self.cfg.action_horizon, self.cfg.action_input_dim)}, "
                f"got {tuple(action_input.shape)}"
            )
        y, y_mask = self._prepare_text(text_embedding, text_mask)
        dtype = self.backbone.patch_embedding.weight.dtype
        device = reference_latent.device
        reference_latent = reference_latent.to(dtype=dtype)
        future_noisy = future_noisy.to(device=device, dtype=dtype)
        if state is None:
            state = torch.zeros(
                b,
                self.cfg.state_tokens,
                self.cfg.state_input_dim,
                device=device,
                dtype=dtype,
            )
        state_tokens = self.state_projector(state.to(device=device, dtype=dtype))
        action_tokens = self.action_projector(action_input.to(device=device, dtype=dtype))
        shared_tokens = torch.cat([state_tokens, action_tokens], dim=1)
        zero_move = torch.zeros(b, 81, device=device, dtype=torch.long)
        # ``memory_is_precomputed`` concatenates image_cond and local_memory.
        # An empty image_cond avoids duplicating the single reference plane.
        empty_history = reference_latent[:, :, :0]
        raw = self.backbone(
            x=future_noisy,
            t=visual_timestep.to(device=device, dtype=torch.float32),
            y=y.to(device=device, dtype=dtype),
            y_mask=y_mask.to(device=device),
            image_cond=empty_history,
            memory_is_precomputed=True,
            local_memory=reference_latent,
            move=zero_move,
            view=zero_move,
            disable_native_action_embedding=True,
            shared_action_tokens=shared_tokens,
            return_shared_action_tokens=True,
            policy_only=not return_video,
            action_timestep=torch.zeros(b, device=device, dtype=torch.float32),
        )
        action_hidden_all = raw["shared_action_hidden"]
        action_hidden = action_hidden_all[:, -self.cfg.action_horizon :]
        self.policy_head.float()
        result = {
            "action_logits": self.policy_head(action_hidden.float()),
            "action_hidden": action_hidden,
            "all_prefix_action_hidden": action_hidden_all,
        }
        if return_video:
            result["video_velocity"] = raw["video"]
        return result

    def flow_sample(self, clean_future: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample the GigaPolicy-0.5 shifted visual flow objective."""

        b = clean_future.shape[0]
        sigma = torch.rand(b, device=clean_future.device, dtype=torch.float32)
        shift = float(self.cfg.visual_flow_shift)
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        sigma = torch.round(sigma * 1000.0) / 1000.0
        noise = torch.randn_like(clean_future)
        sigma_view = sigma.view(b, *([1] * (clean_future.ndim - 1))).to(clean_future.dtype)
        return {
            "noisy": noise * sigma_view + clean_future * (1.0 - sigma_view),
            "target_velocity": noise - clean_future,
            "timestep": sigma * 1000.0,
            "sigma": sigma,
        }

    def multitask_loss(
        self,
        output: dict[str, torch.Tensor],
        *,
        video_target: torch.Tensor | None,
        action_target: torch.Tensor,
        action_mask: torch.Tensor | None,
        include_video: bool,
        include_policy: bool,
        video_weight: float = 1.0,
        policy_weight: float = 5.0,
    ) -> dict[str, torch.Tensor]:
        """Compute independently switchable video-flow and policy-CE losses."""

        zero = output["action_logits"].sum() * 0.0
        video_loss = zero
        if include_video:
            if video_target is None:
                raise ValueError("video_target is required when include_video=True")
            video_loss = F.mse_loss(output["video_velocity"].float(), video_target.float())
        policy_loss = zero
        if include_policy:
            policy_loss = self.loss(output, action_target, action_mask)["loss"]
        total = video_weight * video_loss + policy_weight * policy_loss
        return {
            "loss": total,
            "loss_video_flow": video_loss.detach(),
            "loss_policy_ce": policy_loss.detach(),
            "action_pred": output["action_logits"].argmax(dim=-1),
        }

    def loss(self, output: dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        logits = output["action_logits"]
        target = target.to(device=logits.device, dtype=torch.long)
        if target.shape != logits.shape[:2]:
            raise ValueError(f"target {tuple(target.shape)} != logits prefix {tuple(logits.shape[:2])}")
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none").view_as(target)
        if mask is not None:
            mask = mask.to(device=logits.device, dtype=ce.dtype)
            value = (ce * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            value = ce.mean()
        return {"loss": value, "loss_policy_ce": value.detach(), "action_pred": logits.argmax(dim=-1)}

    def structural_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        backbone = sum(parameter.numel() for parameter in self.backbone.parameters())
        return {
            "model": "GigaNavModel",
            "ablation": "GigaWorld causal policy layout on Wan2.1-T2V-1.3B",
            "parameter_count": total,
            "backbone_parameter_count": backbone,
            "backbone_hidden_dim": self.cfg.hidden_dim,
            "backbone_layers": self.cfg.num_layers,
            "attention_backend": self.attention_backend,
            "giga_causal_attention": self.cfg.giga_causal_attention,
            "obs_structure": "[B,16,4,56,112] T4 latent; first latent is clean reference, last 3 are future slots",
            "token_order": f"[state(1), reference_visual, action({self.cfg.action_horizon}), noisy_future_visual]",
            "text_structure": "UMT5 [B,1,512,4096] through Wan text cross-attention",
            "action_structure": f"zero action query slots [B,{self.cfg.action_horizon},14] -> shared Wan hidden -> Linear -> [B,{self.cfg.action_horizon},4]",
            "policy_only": True,
            "available_objectives": ["policy_only", "video_only", "cotrain"],
            "initialization": "official Wan2.1-1.3B shape-compatible keys; new state/action projectors and nav head fresh",
            "giga_native_deviations": [
                "Wan2.1 native 16-channel latent replaces Wan2.2-5B 48-channel latent",
                "R2R discrete four-class action CE replaces Giga continuous 14-D flow action loss",
                "R2R has no proprioceptive state; state token is an explicit zero vector",
            ],
        }
