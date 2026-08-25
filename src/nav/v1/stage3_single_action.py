"""Single-step closed-loop R2R policy components for Stage3 diagnostics.

This module intentionally leaves the existing H=10 Stage3 implementation
untouched.  It reuses the complete Stage2 Wan/Register model and prepared R2R
T4 observations, but asks one deliberately narrower question: can the shared
hidden state learn one balanced four-way next-action decision?
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nav.v1.stage2_final import FinalStage2WanModel
from nav.v1.stage2_final.data import (
    FINAL_LATENT_H,
    FINAL_LATENT_T,
    FINAL_LATENT_W,
    FINAL_MICRO_FRAMES,
    FINAL_MICRO_STRIDE,
)
from nav.v1.stage3_r2r import (
    ACTION_NAMES,
    MOVE_FORWARD_ACTION,
    R2RPolicyWindow,
    R2RStage3DataConfig,
    R2RStage3PolicyBatchBuilder,
    STOP_ACTION,
    TURN_LEFT_ACTION,
    TURN_RIGHT_ACTION,
    vln_action_to_combo,
)


SINGLE_ACTION_CLASSES = (
    STOP_ACTION,
    MOVE_FORWARD_ACTION,
    TURN_LEFT_ACTION,
    TURN_RIGHT_ACTION,
)
COMBO_TO_ACTION_CLASS = {
    vln_action_to_combo(action_id): action_id for action_id in SINGLE_ACTION_CLASSES
}


@dataclass(slots=True)
class SingleActionPolicyConfig:
    hidden_dim: int = 1536
    mlp_dim: int = 512
    num_classes: int = 4
    policy_queries: int = 1
    history_action_horizon: int = 10
    sampler: str = "balanced_class_cycle"
    action_query_input: str = "zero_noise_zero_timestep"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SingleActionPolicyHead(nn.Module):
    """Read one final Wan action hidden and return four primitive logits."""

    def __init__(self, cfg: SingleActionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.proj_in = nn.Linear(cfg.hidden_dim, cfg.mlp_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(cfg.mlp_dim, cfg.num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1] < self.cfg.policy_queries:
            raise ValueError(f"expected action hidden [B,>=1,C], got {tuple(hidden.shape)}")
        hidden = hidden[:, : self.cfg.policy_queries].float()
        return self.proj_out(self.act(self.proj_in(self.norm(hidden))))


class SingleActionStage3Model(nn.Module):
    """Compose the complete Stage2 world model with a four-way policy head."""

    def __init__(self, world_model: FinalStage2WanModel, cfg: SingleActionPolicyConfig) -> None:
        super().__init__()
        self.world_model = world_model
        self.cfg = cfg
        self.policy_head = SingleActionPolicyHead(cfg).float()

    def keep_policy_modules_fp32(self) -> None:
        """Avoid bf16 update quantization for fresh policy-only parameters."""

        self.policy_head.float()
        self.world_model.action_interface.action_noise_in.float()
        self.world_model.action_interface.action_position.float()
        self.world_model.action_interface.action_timestep.float()

    def policy_fast_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.policy_head,
            self.world_model.action_interface.action_noise_in,
            self.world_model.action_interface.action_position,
            self.world_model.action_interface.action_timestep,
        )
        return [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]

    def backbone_parameters(self) -> list[nn.Parameter]:
        fast_ids = {id(parameter) for parameter in self.policy_fast_parameters()}
        return [
            parameter
            for parameter in self.world_model.parameters()
            if parameter.requires_grad and id(parameter) not in fast_ids
        ]

    def forward_policy(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.world_model.forward_core(
            batch,
            return_prefix_hidden=False,
            include_current_action_condition=False,
        )
        hidden = out.get("shared_action_hidden")
        if hidden is None or hidden.shape[1] != self.cfg.policy_queries:
            raise RuntimeError(
                f"single-action policy requires exactly one shared action hidden, got "
                f"{None if hidden is None else tuple(hidden.shape)}"
            )
        logits = self.policy_head(hidden)
        target = batch["action_class"].to(device=logits.device).long()
        if target.shape != logits.shape[:2]:
            raise RuntimeError(f"target shape {tuple(target.shape)} != logits prefix {tuple(logits.shape[:2])}")
        loss_ce = F.cross_entropy(logits.reshape(-1, self.cfg.num_classes), target.reshape(-1))
        return {
            **out,
            "loss": loss_ce,
            "loss_ce": loss_ce.detach(),
            "action_logits": logits,
            "action_pred": logits.argmax(dim=-1),
        }

    def structural_report(self) -> dict[str, Any]:
        return {
            "wrapper_class": type(self).__name__,
            "world_model": self.world_model.structural_report(),
            "policy_head": type(self.policy_head).__name__,
            "policy_hidden_shape": ["B", 1, self.cfg.hidden_dim],
            "policy_logits_shape": ["B", 1, self.cfg.num_classes],
            "policy_head_parameters": sum(parameter.numel() for parameter in self.policy_head.parameters()),
            "policy_fast_parameter_count": sum(parameter.numel() for parameter in self.policy_fast_parameters()),
            "action_names": {str(key): value for key, value in ACTION_NAMES.items()},
            "policy_config": self.cfg.to_dict(),
        }


class BalancedSingleActionR2RBatchBuilder(R2RStage3PolicyBatchBuilder):
    """Sample one next action with an exact four-class cyclic schedule.

    The underlying latent and text files are never copied.  Sampling with
    replacement is the in-memory equivalent of duplicating minority examples.
    With physical batch size one and gradient accumulation divisible by four,
    every optimizer step receives the same count of all four target classes.
    """

    def __init__(
        self,
        cfg: R2RStage3DataConfig,
        *,
        history_action_horizon: int = 10,
    ) -> None:
        if cfg.action_horizon != 1:
            raise ValueError("single-action builder requires action_horizon=1")
        if cfg.batch_size != 1:
            raise ValueError("balanced single-action builder currently requires physical batch_size=1")
        cfg.action_oversample_mode = "none"
        super().__init__(cfg)
        self.history_action_horizon = int(history_action_horizon)
        self.class_pools: dict[int, list[R2RPolicyWindow]] = {key: [] for key in SINGLE_ACTION_CLASSES}
        for window in self.windows:
            combo = int(self._action_window(self.action_cache[window.sample_id], window.label_start_action)[0])
            action_class = COMBO_TO_ACTION_CLASS.get(combo)
            if action_class is None:
                raise RuntimeError(f"unsupported single-action combo={combo} sample={window.sample_id}")
            self.class_pools[action_class].append(window)
        empty_classes = [ACTION_NAMES[key] for key, values in self.class_pools.items() if not values]
        if empty_classes:
            raise RuntimeError(f"single-action balanced sampler has empty classes: {empty_classes}")
        for values in self.class_pools.values():
            self.rng.shuffle(values)
        self.class_pool_cursors = {key: 0 for key in SINGLE_ACTION_CLASSES}
        self.class_cycle: list[int] = []
        self.class_cycle_cursor = 0
        self._reshuffle_class_cycle()

    def _reshuffle_class_cycle(self) -> None:
        self.class_cycle = list(SINGLE_ACTION_CLASSES)
        self.rng.shuffle(self.class_cycle)
        self.class_cycle_cursor = 0

    def _next_balanced_window(self) -> tuple[int, R2RPolicyWindow]:
        if self.class_cycle_cursor >= len(self.class_cycle):
            self._reshuffle_class_cycle()
        action_class = self.class_cycle[self.class_cycle_cursor]
        self.class_cycle_cursor += 1
        values = self.class_pools[action_class]
        cursor = self.class_pool_cursors[action_class]
        if cursor >= len(values):
            self.rng.shuffle(values)
            cursor = 0
        self.class_pool_cursors[action_class] = cursor + 1
        return action_class, values[cursor]

    @staticmethod
    def _combo_window(actions: list[int], start: int, horizon: int) -> torch.Tensor:
        stop_index = next((index for index, value in enumerate(actions) if int(value) == STOP_ACTION), len(actions) - 1)
        values = []
        for index in range(start, start + horizon):
            action_id = int(actions[index]) if index < len(actions) else STOP_ACTION
            if index > stop_index:
                action_id = STOP_ACTION
            values.append(vln_action_to_combo(action_id))
        return torch.tensor(values, dtype=torch.long)

    def next_batch(self, *, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        action_class, item = self._next_balanced_window()
        payload = self._payload(item.latent_path)
        chunks = payload["micro_latents"][0].float()
        expected = (16, FINAL_LATENT_T, FINAL_LATENT_H, FINAL_LATENT_W)
        if tuple(chunks.shape[1:]) != expected:
            raise RuntimeError(f"bad latent shape for {item.latent_path}: {tuple(chunks.shape)}")
        history = chunks[item.start_micro : item.start_micro + item.history_micro]
        z_obs = chunks[item.obs_micro]
        actions = self._actions_for(payload)
        hist_rows = []
        for micro_index in range(item.start_micro, item.start_micro + item.history_micro):
            hist_start = micro_index * FINAL_MICRO_STRIDE + (FINAL_MICRO_FRAMES - 1)
            hist_rows.append(self._combo_window(actions, hist_start, self.history_action_horizon))
        target_combo = self._combo_window(actions, item.label_start_action, 1)
        expected_combo = vln_action_to_combo(action_class)
        if int(target_combo[0]) != expected_combo:
            raise RuntimeError(
                f"balanced class/target mismatch class={action_class} combo={int(target_combo[0])}"
            )
        y, y_mask, text_source = self._text(item.dataset, item.sample_id)
        z_obs_batch = z_obs[None].to(device=device, dtype=dtype, non_blocking=True)
        batch = {
            "history_latents": history[None].to(device=device, dtype=dtype, non_blocking=True),
            "z_obs": z_obs_batch,
            "z_future_noisy": torch.zeros_like(z_obs_batch),
            "visual_timestep": torch.zeros(1, device=device, dtype=torch.float32),
            "a_hist_combo": torch.stack(hist_rows, dim=0)[None].to(device=device, non_blocking=True),
            "a_cur_combo": torch.zeros(1, 1, device=device, dtype=torch.long),
            "a_noise": torch.zeros(1, 1, 6, device=device, dtype=dtype),
            "action_timestep": torch.zeros(1, device=device, dtype=torch.float32),
            "action_combo": target_combo[None].to(device=device, non_blocking=True),
            "action_class": torch.tensor([[action_class]], device=device, dtype=torch.long),
            "action_loss_mask": torch.ones(1, 1, device=device, dtype=torch.float32),
            "action_target": torch.zeros(1, 1, 6, device=device, dtype=dtype),
            "y": y.to(device=device, dtype=dtype, non_blocking=True),
            "y_mask": y_mask.to(device=device, dtype=dtype, non_blocking=True),
        }
        meta = {
            "history_micro": item.history_micro,
            "sample_ids": [item.sample_id],
            "label_start_actions": [item.label_start_action],
            "target_action_class": action_class,
            "target_action_name": ACTION_NAMES[action_class],
            "terminal_count": int(item.is_terminal),
            "action_valid": 1.0,
            "text_hits": self.text_hits,
            "text_fallbacks": self.text_fallbacks,
            "text_sources": [text_source],
        }
        return batch, meta

    def summary(self) -> dict[str, Any]:
        payload = super().summary()
        natural_counts = Counter({ACTION_NAMES[key]: len(values) for key, values in self.class_pools.items()})
        total = sum(natural_counts.values())
        payload["single_action"] = {
            "policy_horizon": 1,
            "history_action_horizon": self.history_action_horizon,
            "sampler": "balanced four-class cycle with replacement; no latent duplication",
            "natural_window_counts": dict(natural_counts),
            "natural_window_ratio": {
                key: float(value) / float(total) for key, value in natural_counts.items()
            },
            "effective_sample_ratio": {ACTION_NAMES[key]: 0.25 for key in SINGLE_ACTION_CLASSES},
            "target_shape": ["B", 1],
        }
        return payload
