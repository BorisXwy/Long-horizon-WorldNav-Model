"""Config deserialization shared by training, inference, and evaluation."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from nav.v1.data.r2r import R2RStage3DataConfig
from nav.v1.data.stage2 import FinalStage2DataConfig


ConfigT = TypeVar("ConfigT", FinalStage2DataConfig, R2RStage3DataConfig)
_PATH_FIELDS = {
    "latent_manifest_dir",
    "latent_root",
    "manifest",
    "re10k_camera_root",
    "rendered_manifest",
    "text_cache_root",
    "text_empty",
}


def data_config_from_dict(cls: type[ConfigT], payload: dict[str, Any]) -> ConfigT:
    """Restore a known V1 data dataclass while preserving saved field values."""

    allowed = {field.name for field in fields(cls)}
    values = {key: value for key, value in payload.items() if key in allowed}
    for key in _PATH_FIELDS.intersection(values):
        values[key] = Path(values[key])
    return cls(**values)


def stage2_data_config_from_dict(payload: dict[str, Any]) -> FinalStage2DataConfig:
    return data_config_from_dict(FinalStage2DataConfig, payload)


def r2r_data_config_from_dict(payload: dict[str, Any]) -> R2RStage3DataConfig:
    return data_config_from_dict(R2RStage3DataConfig, payload)


def stage2_data_config_from_checkpoint(payload: dict[str, Any]) -> FinalStage2DataConfig:
    config = payload.get("data_config")
    if config is None:
        config = payload.get("stage2_replay_data_config")
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint has neither data_config nor stage2_replay_data_config")
    return stage2_data_config_from_dict(config)
