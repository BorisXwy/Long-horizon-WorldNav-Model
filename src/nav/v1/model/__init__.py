"""Canonical V1 model modules and central assembly API."""

from .builder import (
    ModelAssembly,
    assemble_stage2_checkpoint,
    assemble_stage2_wan_init,
    assemble_stage3_single_action_checkpoint,
    available_model_assemblies,
    build_model,
    register_model_assembler,
)
from .policy import SingleActionPolicyConfig, SingleActionPolicyHead, SingleActionStage3Model
from .world_model import FinalStage2WanConfig, FinalStage2WanModel

__all__ = [
    "FinalStage2WanConfig",
    "FinalStage2WanModel",
    "ModelAssembly",
    "SingleActionPolicyConfig",
    "SingleActionPolicyHead",
    "SingleActionStage3Model",
    "assemble_stage2_checkpoint",
    "assemble_stage2_wan_init",
    "assemble_stage3_single_action_checkpoint",
    "available_model_assemblies",
    "build_model",
    "register_model_assembler",
]
