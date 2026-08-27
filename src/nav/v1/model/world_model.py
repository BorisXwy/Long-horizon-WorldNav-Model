"""Canonical shared Wan/Register world-model modules for V1 Stage2/Stage3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[5]
_IW_ROOT = Path(os.environ.get("NAV_INF_WORLD_ROOT", str(_PROJECT_ROOT / "Infinite-World")))
if str(_IW_ROOT) not in sys.path:
    sys.path.insert(0, str(_IW_ROOT))

from infworld.models.dit_model import WanModel  # noqa: E402

from nav.v1.models.heads import DenseCameraQueryPoseHead, FramePoseHead  # noqa: E402
from nav.v1.models.iw_aligned import (  # noqa: E402
    IWActionInterface,
    IWAlignedConfig,
    SpatialRegisterMemory,
    install_attention_fallback_if_needed,
)


install_attention_fallback_if_needed()


@dataclass(slots=True)
class FinalStage2WanConfig:
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
    pose_head_type: str = "frame_mlp"
    pose_readout_layer: int = -1
    register_mode: str = "fixed_rnull_unified"
    register_injection: str = "condition_memory"
    register_condition_grid: tuple[int, int, int] = (4, 4, 8)
    register_grad_tail: int = 4
    z_obs_stream: str = "main_prefix_clean_obs"
    action_condition_injection: str = "shared_condition_tokens"
    use_channel_mask: bool = False
    policy_head_type: str = "iw_flow_combo"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_iw_aligned_config(self) -> IWAlignedConfig:
        return IWAlignedConfig(
            latent_channels=self.latent_channels,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            ffn_dim=self.ffn_dim,
            freq_dim=self.freq_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            caption_channels=self.caption_channels,
            model_max_length=self.model_max_length,
            register_frames=self.register_frames,
            register_hidden_dim=self.register_hidden_dim,
            register_heads=self.register_heads,
            register_layers=self.register_layers,
            chunk_time_tokens=self.chunk_time_tokens,
            action_horizon=self.action_horizon,
            action_vocab_size=self.action_vocab_size,
            combo_action_vocab_size=self.combo_action_vocab_size,
            pose_head_variant=self.pose_head_variant,
            include_current_obs_prefix=True,
            update_register_with_current=False,
            pose_probe_mode="main_prefix_current_obs",
            register_mode=self.register_mode,
            current_action_mode="none",
        )


class FinalStage2WanModel(nn.Module):
    """Real Wan2.1/IW DiT with final Stage2 token semantics.

    Main stream for this run:
      default ``condition_memory`` mode uses
      ``Z_obs + A_noise + Z_future_noise`` inside Wan's shared token stream.
      Wan internally inserts action tokens after the clean visual prefix and
      before future visual tokens.  In the explicit ``main_prefix`` ablation,
      ``Register R_t`` is also prepended to the clean visual prefix.

    Condition stream:
      text context through Wan text cross-attention; ``Register R_t`` and
      ``A_cur`` enter through ``shared_condition_tokens`` in the default
      ``condition_memory`` mode.  ``A_hist`` is only consumed before the
      backbone by the Register updater.
    """

    def __init__(self, cfg: FinalStage2WanConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FinalStage2WanConfig()
        if self.cfg.register_injection not in {"main_prefix", "condition_memory"}:
            raise ValueError("register_injection must be main_prefix or condition_memory")
        if self.cfg.policy_head_type not in {"iw_flow_combo", "linear"}:
            raise ValueError("policy_head_type must be iw_flow_combo or linear")
        iw_cfg = self.cfg.to_iw_aligned_config()
        self.iw_cfg = iw_cfg
        self.backbone = WanModel(
            model_type="t2v",
            dim=iw_cfg.hidden_dim,
            in_channels=iw_cfg.latent_channels,
            ffn_dim=iw_cfg.ffn_dim,
            freq_dim=iw_cfg.freq_dim,
            num_heads=iw_cfg.num_heads,
            num_layers=iw_cfg.num_layers,
            out_channels=iw_cfg.latent_channels,
            caption_channels=iw_cfg.caption_channels,
            model_max_length=iw_cfg.model_max_length,
            use_convenc=True,
        )
        self.register_memory = SpatialRegisterMemory(iw_cfg)
        self.action_interface = IWActionInterface(iw_cfg)
        self.policy_linear_head = (
            nn.Linear(iw_cfg.hidden_dim, iw_cfg.combo_action_vocab_size)
            if self.cfg.policy_head_type == "linear"
            else None
        )
        if self.policy_linear_head is not None:
            self._freeze_unused_action_output_heads()
        self.register_condition_proj = nn.Sequential(
            nn.LayerNorm(iw_cfg.latent_channels),
            nn.Linear(iw_cfg.latent_channels, iw_cfg.hidden_dim),
        )
        if self.cfg.pose_head_type == "frame_mlp":
            self.pose_head = FramePoseHead(
                hidden_dim=iw_cfg.hidden_dim,
                patch_size=(1, 2, 2),
                variant=iw_cfg.pose_head_variant,
            )
        elif self.cfg.pose_head_type == "dense_camera_query":
            self.pose_head = DenseCameraQueryPoseHead(
                hidden_dim=iw_cfg.hidden_dim,
                patch_size=(1, 2, 2),
                num_heads=8,
            )
        else:
            raise ValueError(f"unsupported pose_head_type: {self.cfg.pose_head_type}")
        self._pose_layer_hidden: torch.Tensor | None = None
        self._pose_layer_hook = None
        if self.cfg.pose_readout_layer >= 0:
            max_layer = len(self.backbone.blocks) - 1
            if self.cfg.pose_readout_layer > max_layer:
                raise ValueError(f"pose_readout_layer {self.cfg.pose_readout_layer} > max layer {max_layer}")
            self._pose_layer_hook = self.backbone.blocks[self.cfg.pose_readout_layer].register_forward_hook(
                self._capture_pose_layer_hidden
            )
        self.hmpc_removed = False

    def _freeze_unused_action_output_heads(self) -> None:
        """Disable legacy flow/classification decoders in linear-policy mode."""

        for module in (
            self.action_interface.action_out_norm,
            self.action_interface.action_velocity_head,
            self.action_interface.primitive_head,
            self.action_interface.combo_head,
            self.action_interface.move_head,
            self.action_interface.view_head,
        ):
            module.requires_grad_(False)

    def enable_linear_policy_head(self) -> dict[str, Any]:
        """Attach a fresh Linear(hidden_dim -> 144) discrete policy head.

        This is called only after a Stage2 checkpoint has loaded, so the full
        shared backbone/Register/video/pose state remains intact while the
        Stage3-only classifier starts from a clean initialization.
        """

        if self.policy_linear_head is None:
            reference = self.backbone.patch_embedding.weight
            self.policy_linear_head = nn.Linear(
                self.cfg.hidden_dim,
                self.cfg.combo_action_vocab_size,
            ).to(device=reference.device, dtype=reference.dtype)
        self.cfg.policy_head_type = "linear"
        self._freeze_unused_action_output_heads()
        return {
            "policy_head_type": "linear",
            "input_dim": self.cfg.hidden_dim,
            "output_dim": self.cfg.combo_action_vocab_size,
            "parameters": sum(parameter.numel() for parameter in self.policy_linear_head.parameters()),
            "initialization": "fresh_after_stage2_checkpoint_load",
            "legacy_output_heads_frozen": True,
        }

    def _capture_pose_layer_hidden(self, _module, _inputs, output) -> None:
        if isinstance(output, tuple):
            output = output[0]
        self._pose_layer_hidden = output

    def remove_hmpc(self) -> None:
        self.backbone.latent_encoder = nn.Identity()
        self.backbone.use_convenc = False
        self.hmpc_removed = True

    def load_wan_checkpoint(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if path.suffix == ".safetensors":
            raw = load_file(str(path), device="cpu")
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            raw = payload.get("state_dict", payload)
        target = self.backbone.state_dict()
        state = {k: v for k, v in raw.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
        result = self.backbone.load_state_dict(state, strict=False)
        self.remove_hmpc()
        return {
            "path": str(path),
            "infworld_root": str(_IW_ROOT),
            "use_channel_mask": False,
            "loaded_keys": len(state),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "skipped_shape_mismatch": [
                {"key": k, "source": list(v.shape), "target": list(target[k].shape)}
                for k, v in raw.items()
                if k in target and tuple(v.shape) != tuple(target[k].shape)
            ][:64],
        }

    def _t_for_wan(self, visual_timestep: torch.Tensor) -> torch.Tensor:
        t = visual_timestep.float()
        if float(t.detach().max().cpu()) <= 1.5:
            t = t * 1000.0
        return t

    def roll_register(self, history_latents: torch.Tensor, a_hist_combo: torch.Tensor) -> torch.Tensor:
        """Roll the full history into Register with truncated BPTT.

        The formal sample still consumes every history chunk.  To make IW16
        equivalent windows trainable, early history chunks are used as state
        updates under ``no_grad``; the final ``register_grad_tail`` updates keep
        gradients for the Register updater/extractor and downstream DiT.
        """
        dtype = self.backbone.patch_embedding.weight.dtype
        steps = int(history_latents.shape[1])
        grad_tail = max(0, int(self.cfg.register_grad_tail))
        grad_start = max(0, steps - grad_tail)

        if grad_start > 0:
            with torch.no_grad():
                registers = self.register_memory.extract(
                    history_latents[:, 0],
                    action_context=self.action_interface.hist_context_tokens(a_hist_combo[:, 0], dtype=dtype),
                )
                for index in range(1, grad_start):
                    registers = self.register_memory.update(
                        registers,
                        history_latents[:, index],
                        action_context=self.action_interface.hist_context_tokens(a_hist_combo[:, index], dtype=dtype),
                    )
            registers = registers.detach()
            start = grad_start
        else:
            registers = self.register_memory.extract(
                history_latents[:, 0],
                action_context=self.action_interface.hist_context_tokens(a_hist_combo[:, 0], dtype=dtype),
            )
            start = 1

        for index in range(start, steps):
            registers = self.register_memory.update(
                registers,
                history_latents[:, index],
                action_context=self.action_interface.hist_context_tokens(a_hist_combo[:, index], dtype=dtype),
            )
        return registers

    def roll_register_variable_history(
        self,
        history_latents: torch.Tensor,
        a_hist_combo: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Roll padded full prefixes without letting padding update Register.

        Each sample is rolled independently so ``register_grad_tail`` remains
        relative to that sample's real prefix length.  Only the inexpensive
        recurrent Register path is looped per sample; the shared Wan backbone
        still receives the complete physical batch in one forward.
        """

        if history_latents.ndim != 6:
            raise ValueError(f"expected padded history [B,K,C,T,H,W], got {tuple(history_latents.shape)}")
        if a_hist_combo.ndim < 3 or a_hist_combo.shape[:2] != history_latents.shape[:2]:
            raise ValueError(
                f"history/action prefix mismatch: history={tuple(history_latents.shape)} "
                f"actions={tuple(a_hist_combo.shape)}"
            )
        lengths = history_lengths.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
        if len(lengths) != history_latents.shape[0]:
            raise ValueError(f"history_lengths={lengths} batch={history_latents.shape[0]}")
        registers = []
        for batch_index, length in enumerate(lengths):
            length = int(length)
            if length <= 0 or length > history_latents.shape[1]:
                raise ValueError(f"invalid history length {length}; padded K={history_latents.shape[1]}")
            registers.append(
                self.roll_register(
                    history_latents[batch_index : batch_index + 1, :length],
                    a_hist_combo[batch_index : batch_index + 1, :length],
                )
            )
        return torch.cat(registers, dim=0)

    def current_obs_hidden(self, prefix_video_hidden: torch.Tensor, z_obs: torch.Tensor) -> torch.Tensor:
        h_tokens = z_obs.shape[-2] // 2
        w_tokens = z_obs.shape[-1] // 2
        register_tokens = self.cfg.register_frames * h_tokens * w_tokens if self.cfg.register_injection == "main_prefix" else 0
        obs_tokens = z_obs.shape[2] * h_tokens * w_tokens
        return prefix_video_hidden[:, register_tokens : register_tokens + obs_tokens]

    def register_condition_tokens(self, registers: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(registers.float(), self.cfg.register_condition_grid)
        tokens = pooled.permute(0, 2, 3, 4, 1).reshape(registers.shape[0], -1, registers.shape[1])
        tokens = tokens.to(dtype=next(self.register_condition_proj.parameters()).dtype)
        return self.register_condition_proj(tokens)

    @staticmethod
    def video_flow_target(batch: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        return batch["z_future_noise"].to(dtype) - batch["z_future_target"].to(dtype)

    def forward_core(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_prefix_hidden: bool = True,
        include_current_action_condition: bool = True,
    ) -> dict[str, Any]:
        if not self.hmpc_removed:
            raise RuntimeError("load_wan_checkpoint() must be called before training")
        self._pose_layer_hidden = None
        dtype = self.backbone.patch_embedding.weight.dtype
        history = batch["history_latents"].to(dtype)
        z_obs = batch["z_obs"].to(dtype)
        z_future_noisy = batch["z_future_noisy"].to(dtype)
        b = z_obs.shape[0]
        a_hist_combo = batch["a_hist_combo"].to(device=z_obs.device)
        history_lengths = batch.get("history_lengths")
        if history_lengths is None:
            registers = self.roll_register(history, a_hist_combo)
        else:
            registers = self.roll_register_variable_history(history, a_hist_combo, history_lengths)
        if self.cfg.register_injection == "main_prefix":
            image_cond = torch.cat([registers.to(dtype), z_obs], dim=2)
            register_condition = None
        else:
            image_cond = z_obs
            register_condition = self.register_condition_tokens(registers).to(device=z_obs.device, dtype=dtype)
        local_memory = z_obs[:, :, -1:].contiguous()
        y = batch["y"].to(device=z_obs.device, dtype=dtype)
        y_mask = batch["y_mask"].to(device=z_obs.device, dtype=dtype)
        action_timestep = batch.get("action_timestep", torch.zeros(b, device=z_obs.device, dtype=torch.float32))
        action_tokens = self.action_interface.noise_tokens(
            batch["a_noise"].to(device=z_obs.device),
            action_timestep.to(device=z_obs.device),
            dtype=dtype,
        )
        if include_current_action_condition:
            a_cur_condition = self.action_interface.current_condition(batch["a_cur_combo"].to(device=z_obs.device)).to(
                device=z_obs.device,
                dtype=dtype,
            )
            video_condition = (
                torch.cat([register_condition, a_cur_condition], dim=1)
                if register_condition is not None
                else a_cur_condition
            )
        else:
            video_condition = register_condition
        policy_condition = register_condition
        noop = torch.zeros(b, 81, device=z_obs.device, dtype=torch.long)
        out = self.backbone(
            x=z_future_noisy,
            t=self._t_for_wan(batch["visual_timestep"].to(z_obs.device)),
            y=y,
            y_mask=y_mask,
            image_cond=image_cond,
            memory_is_precomputed=True,
            local_memory=local_memory,
            move=noop,
            view=noop,
            disable_native_action_embedding=True,
            shared_action_tokens=action_tokens,
            shared_condition_tokens=video_condition,
            shared_policy_condition_tokens=policy_condition,
            return_shared_action_tokens=True,
            return_prefix_video_hidden=return_prefix_hidden,
            action_timestep=action_timestep.to(z_obs.device),
        )
        video = out["video"] if isinstance(out, dict) else out
        if video.shape[2] != z_future_noisy.shape[2]:
            video = video[:, :, -z_future_noisy.shape[2] :].contiguous()
        shared_action_hidden = out.get("shared_action_hidden") if isinstance(out, dict) else None
        prefix_hidden = out.get("prefix_video_hidden") if isinstance(out, dict) else None
        if self.cfg.policy_head_type == "linear":
            if shared_action_hidden is None or self.policy_linear_head is None:
                raise RuntimeError("linear policy head requires shared action hidden tokens")
            action_outputs = {
                "action_hidden": shared_action_hidden[:, : self.cfg.action_horizon],
                "combo_logits": self.policy_linear_head(
                    shared_action_hidden[:, : self.cfg.action_horizon].to(self.policy_linear_head.weight.dtype)
                ),
            }
        else:
            action_outputs = self.action_interface.decode(shared_action_hidden)
        return {
            "future_velocity": video,
            "shared_action_hidden": shared_action_hidden,
            "action_outputs": action_outputs,
            "prefix_video_hidden": prefix_hidden,
            "pose_layer_hidden": self._pose_layer_hidden,
            "registers": registers,
        }

    def pose_obs_hidden(self, out: dict[str, Any], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.cfg.pose_readout_layer >= 0:
            hidden = out.get("pose_layer_hidden")
            if hidden is None:
                raise RuntimeError(f"pose_readout_layer={self.cfg.pose_readout_layer} did not produce hidden states")
            return self.current_obs_hidden(hidden, batch["z_obs"])
        prefix = out.get("prefix_video_hidden")
        if prefix is None:
            raise RuntimeError("prefix_video_hidden is required for default pose readout")
        return self.current_obs_hidden(prefix, batch["z_obs"])

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        target = target.to(device=pred.device, dtype=pred.dtype)
        if mask is None:
            return F.mse_loss(pred, target)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        while mask.ndim < pred.ndim:
            mask = mask.unsqueeze(-1)
        return (((pred - target) ** 2) * mask).sum() / mask.expand_as(pred).sum().clamp_min(1.0)

    def forward_stage2(self, batch: dict[str, torch.Tensor], *, lambda_pose: float) -> dict[str, torch.Tensor]:
        out = self.forward_core(batch, return_prefix_hidden=True)
        future_velocity = out["future_velocity"]
        visual_target = self.video_flow_target(batch, future_velocity.dtype)
        visual_loss = F.mse_loss(future_velocity, visual_target)
        if lambda_pose > 0:
            obs_hidden = self.pose_obs_hidden(out, batch)
            obs_hidden = obs_hidden.to(next(self.pose_head.parameters()).dtype)
            latent_shape = (batch["z_obs"].shape[2], batch["z_obs"].shape[3], batch["z_obs"].shape[4])
            pose_pred = self.pose_head(obs_hidden, latent_shape=latent_shape)
            pose_loss = self._masked_mse(pose_pred, batch["pose_target"], batch.get("pose_mask"))
        else:
            obs_hidden = None
            pose_pred = None
            pose_loss = torch.zeros((), device=visual_loss.device, dtype=visual_loss.dtype)
        loss = visual_loss + float(lambda_pose) * pose_loss
        return {
            **out,
            "obs_hidden": obs_hidden,
            "pose_pred": pose_pred,
            "loss": loss,
            "loss_visual_tensor": visual_loss,
            "loss_pose_tensor": pose_loss,
            "loss_visual": visual_loss.detach(),
            "loss_pose": pose_loss.detach(),
            "loss_3d": pose_loss.detach(),
        }

    def forward_stage3_policy(self, batch: dict[str, torch.Tensor], *, lambda_ce: float = 1.0) -> dict[str, torch.Tensor]:
        """Final Stage3 policy/action loss on the shared Wan action tokens.

        Policy mode uses ``Register + Z_obs + instruction/text + A_noise`` and
        supervises the shared action-token output.  When categorical combo
        labels are present, combo CE is the primary Stage3 loss; continuous
        action flow stays as an auxiliary output for future ablations.
        """

        out = self.forward_core(
            batch,
            return_prefix_hidden=False,
            include_current_action_condition=False,
        )
        action_outputs = out["action_outputs"]
        if action_outputs is None:
            raise RuntimeError("forward_stage3_policy requires shared action hidden")
        pred_velocity = action_outputs.get("action_velocity")
        if "action_combo" in batch:
            combo_logits = action_outputs["combo_logits"]
            loss_action_flow = torch.zeros((), device=combo_logits.device, dtype=combo_logits.dtype)
            combo_target = batch["action_combo"].to(device=combo_logits.device).long()
            if combo_target.ndim == 1:
                combo_logits = combo_logits[:, 0]
            ce_all = F.cross_entropy(
                combo_logits.reshape(-1, combo_logits.shape[-1]),
                combo_target.reshape(-1),
                reduction="none",
            ).view_as(combo_target)
        else:
            if pred_velocity is None:
                raise RuntimeError("continuous action targets require the legacy flow action head")
            action_target = batch["action_target"].to(device=pred_velocity.device, dtype=pred_velocity.dtype)
            if action_target.ndim == 2:
                pred_velocity = pred_velocity[:, 0]
            if pred_velocity.shape != action_target.shape:
                raise RuntimeError(
                    f"action velocity shape {tuple(pred_velocity.shape)} != target {tuple(action_target.shape)}"
                )
            target_velocity = action_target - batch["a_noise"].to(device=pred_velocity.device, dtype=pred_velocity.dtype)
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
            "primitive_logits": action_outputs.get("primitive_logits"),
            "combo_logits": action_outputs["combo_logits"],
        }

    def structural_report(self) -> dict[str, Any]:
        return {
            "model_class": type(self).__name__,
            "backbone_class": type(self.backbone).__name__,
            "infworld_root": str(_IW_ROOT),
            "single_shared_wan_blocks": True,
            "register_injection": self.cfg.register_injection,
            "register_condition_grid": list(self.cfg.register_condition_grid),
            "register_condition_tokens": int(torch.tensor(self.cfg.register_condition_grid).prod().item()),
            "z_obs_stream": self.cfg.z_obs_stream,
            "a_cur_injection": self.cfg.action_condition_injection,
            "a_noise_path": "shared_action_tokens_main_stream",
            "policy_head_type": self.cfg.policy_head_type,
            "policy_query_path": "unchanged_action_noise_plus_timestep",
            "policy_head_parameters": (
                sum(parameter.numel() for parameter in self.policy_linear_head.parameters())
                if self.policy_linear_head is not None
                else sum(parameter.numel() for parameter in self.action_interface.combo_head.parameters())
            ),
            "pose_head_type": self.cfg.pose_head_type,
            "pose_readout_layer": self.cfg.pose_readout_layer,
            "branch_mask_rule": "A_noise and Z_future_noise are mutually isolated in Wan self-attention",
            "policy_condition_rule": "A_noise/action tokens read Register-only policy condition; A_cur is video-only",
            "use_channel_mask": False,
            "config": self.cfg.to_dict(),
        }
