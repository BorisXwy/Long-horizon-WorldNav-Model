"""Discrete action-chunk policy modules assembled on the shared V1 world model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nav.v1.data.r2r import ACTION_NAMES
from nav.v1.model.world_model import FinalStage2WanModel


@dataclass(slots=True)
class SingleActionPolicyConfig:
    hidden_dim: int = 1536
    mlp_dim: int = 512
    num_classes: int = 4
    policy_queries: int = 1
    backbone_action_tokens: int | None = None
    history_action_horizon: int = 10
    sampler: str = "balanced_class_cycle"
    action_query_input: str = "zero_noise_zero_timestep"

    def __post_init__(self) -> None:
        if self.backbone_action_tokens is None:
            self.backbone_action_tokens = self.policy_queries
        if self.backbone_action_tokens < self.policy_queries:
            raise ValueError("backbone_action_tokens must be >= policy_queries")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SingleActionPolicyHead(nn.Module):
    """Decode each selected final Wan action hidden into four primitive logits."""

    def __init__(self, cfg: SingleActionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.proj_in = nn.Linear(cfg.hidden_dim, cfg.mlp_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(cfg.mlp_dim, cfg.num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1] < self.cfg.backbone_action_tokens:
            raise ValueError(
                f"expected action hidden [B,>={self.cfg.backbone_action_tokens},C], got {tuple(hidden.shape)}"
            )
        hidden = hidden[:, : self.cfg.backbone_action_tokens].float()
        hidden = hidden[:, : self.cfg.policy_queries]
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
        if hidden is None or hidden.shape[1] != self.cfg.backbone_action_tokens:
            raise RuntimeError(
                f"policy requires exactly {self.cfg.backbone_action_tokens} shared action hidden tokens, got "
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
            "backbone_action_hidden_shape": ["B", self.cfg.backbone_action_tokens, self.cfg.hidden_dim],
            "policy_hidden_shape": ["B", self.cfg.policy_queries, self.cfg.hidden_dim],
            "policy_readout": (
                f"select hidden slots [0:{self.cfg.policy_queries}] from the unchanged "
                f"{self.cfg.backbone_action_tokens}-slot backbone, then apply one shared MLP"
            ),
            "unused_action_hidden_slots": self.cfg.backbone_action_tokens - self.cfg.policy_queries,
            "policy_logits_shape": ["B", self.cfg.policy_queries, self.cfg.num_classes],
            "policy_head_parameters": sum(parameter.numel() for parameter in self.policy_head.parameters()),
            "policy_fast_parameter_count": sum(parameter.numel() for parameter in self.policy_fast_parameters()),
            "action_names": {str(key): value for key, value in ACTION_NAMES.items()},
            "policy_config": self.cfg.to_dict(),
        }
