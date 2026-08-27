"""Central model assembly entrypoints for formal V1 training and inference."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from nav.v1.model.policy import SingleActionPolicyConfig, SingleActionStage3Model
from nav.v1.model.world_model import FinalStage2WanConfig, FinalStage2WanModel


@dataclass(slots=True)
class ModelAssembly:
    """A constructed model plus auditable initialization metadata."""

    model: nn.Module
    source_payload: dict[str, Any] | None
    audit: dict[str, Any]


def _model_config(payload: dict[str, Any]) -> FinalStage2WanConfig:
    allowed = {field.name for field in fields(FinalStage2WanConfig)}
    values = {key: value for key, value in payload.items() if key in allowed}
    if "register_condition_grid" in values:
        values["register_condition_grid"] = tuple(values["register_condition_grid"])
    return FinalStage2WanConfig(**values)


def _policy_config(payload: dict[str, Any]) -> SingleActionPolicyConfig:
    allowed = {field.name for field in fields(SingleActionPolicyConfig)}
    return SingleActionPolicyConfig(**{key: value for key, value in payload.items() if key in allowed})


def _apply_runtime(
    model: nn.Module,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
    training: bool,
) -> nn.Module:
    if device is not None or dtype is not None:
        model.to(device=device, dtype=dtype)
    model.train(training)
    return model


def assemble_stage2_wan_init(
    *,
    config: FinalStage2WanConfig | None = None,
    checkpoint: str | Path,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    training: bool = True,
) -> ModelAssembly:
    """Build the formal Stage2 graph and initialize it from raw Wan weights."""

    model = FinalStage2WanModel(config)
    audit = model.load_wan_checkpoint(checkpoint)
    _apply_runtime(model, device=device, dtype=dtype, training=training)
    return ModelAssembly(model=model, source_payload=None, audit=audit)


def assemble_stage2_checkpoint(
    *,
    checkpoint: str | Path,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    training: bool = False,
) -> ModelAssembly:
    """Restore a Stage2-compatible world model without changing state keys."""

    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload or "model_config" not in payload:
        raise RuntimeError(f"invalid V1 checkpoint payload: {checkpoint}")
    model = FinalStage2WanModel(_model_config(payload["model_config"]))
    model.remove_hmpc()
    result = model.load_state_dict(payload["model"], strict=False)
    bad_missing = [key for key in result.missing_keys if not key.startswith("backbone.latent_encoder.")]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={bad_missing[:20]} unexpected={result.unexpected_keys[:20]}"
        )
    model._freeze_unused_action_output_heads()
    _apply_runtime(model, device=device, dtype=dtype, training=training)
    return ModelAssembly(
        model=model,
        source_payload=payload,
        audit={
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(payload.get("step", -1)),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "checkpoint_compatible": True,
        },
    )


def assemble_stage3_single_action_checkpoint(
    *,
    checkpoint: str | Path,
    policy_config: SingleActionPolicyConfig | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    training: bool = False,
    require_policy_head: bool = False,
) -> ModelAssembly:
    """Build Stage3 from either a Stage2 source or a Stage3 policy checkpoint."""

    world_assembly = assemble_stage2_checkpoint(
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        training=training,
    )
    world_model = world_assembly.model
    if not isinstance(world_model, FinalStage2WanModel):
        raise TypeError(type(world_model))
    payload = world_assembly.source_payload or {}
    if policy_config is None:
        policy_payload = payload.get("single_action_policy_config")
        policy_config = (
            _policy_config(policy_payload)
            if isinstance(policy_payload, dict)
            else SingleActionPolicyConfig(hidden_dim=world_model.cfg.hidden_dim)
        )
    model = SingleActionStage3Model(world_model, policy_config)
    _apply_runtime(model, device=device, dtype=None, training=training)
    policy_state = payload.get("single_action_policy_head")
    if policy_state is not None:
        model.policy_head.load_state_dict(policy_state, strict=True)
    elif require_policy_head:
        raise RuntimeError(f"checkpoint has no single_action_policy_head: {checkpoint}")
    model.keep_policy_modules_fp32()
    return ModelAssembly(
        model=model,
        source_payload=payload,
        audit={
            **world_assembly.audit,
            "assembly": "stage3_single_action",
            "policy_head_loaded": policy_state is not None,
            "policy_head_initialization": "checkpoint" if policy_state is not None else "fresh",
            "policy_config": policy_config.to_dict(),
        },
    )


ModelAssembler = Callable[..., ModelAssembly]
_MODEL_ASSEMBLERS: dict[str, ModelAssembler] = {
    "stage2_wan_init": assemble_stage2_wan_init,
    "stage2_checkpoint": assemble_stage2_checkpoint,
    "stage3_single_action_checkpoint": assemble_stage3_single_action_checkpoint,
}


def register_model_assembler(name: str, assembler: ModelAssembler, *, replace: bool = False) -> None:
    """Register an experimental assembly without editing existing model modules."""

    if name in _MODEL_ASSEMBLERS and not replace:
        raise KeyError(f"model assembler already registered: {name}")
    _MODEL_ASSEMBLERS[name] = assembler


def build_model(assembly: str, **kwargs: Any) -> ModelAssembly:
    """Build a registered model assembly through the single public entrypoint."""

    try:
        assembler = _MODEL_ASSEMBLERS[assembly]
    except KeyError as exc:
        raise KeyError(f"unknown model assembly {assembly!r}; available={sorted(_MODEL_ASSEMBLERS)}") from exc
    return assembler(**kwargs)


def available_model_assemblies() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_ASSEMBLERS))
