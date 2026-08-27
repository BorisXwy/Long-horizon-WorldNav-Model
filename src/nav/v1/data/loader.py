"""Central data-pipeline assembly entrypoint for formal V1 runs."""

from __future__ import annotations

from typing import Any, Callable

from nav.v1.data.r2r import R2RStage3DataConfig, R2RStage3PolicyBatchBuilder
from nav.v1.data.r2r_full_history import FullHistoryBalancedSingleActionR2RBatchBuilder
from nav.v1.data.sampler import BalancedSingleActionR2RBatchBuilder
from nav.v1.data.stage2 import FinalStage2BatchBuilder, FinalStage2DataConfig


def _stage2_loader(*, config: FinalStage2DataConfig, **_: Any) -> FinalStage2BatchBuilder:
    return FinalStage2BatchBuilder(config)


def _stage3_r2r_loader(*, config: R2RStage3DataConfig, **_: Any) -> R2RStage3PolicyBatchBuilder:
    return R2RStage3PolicyBatchBuilder(config)


def _stage3_single_action_balanced_loader(
    *,
    config: R2RStage3DataConfig,
    history_action_horizon: int = 10,
    **_: Any,
) -> BalancedSingleActionR2RBatchBuilder:
    return BalancedSingleActionR2RBatchBuilder(
        config,
        history_action_horizon=history_action_horizon,
    )


def _stage3_single_action_full_history_loader(
    *,
    config: R2RStage3DataConfig,
    history_action_horizon: int = 10,
    **_: Any,
) -> FullHistoryBalancedSingleActionR2RBatchBuilder:
    return FullHistoryBalancedSingleActionR2RBatchBuilder(
        config,
        history_action_horizon=history_action_horizon,
    )


DataLoaderFactory = Callable[..., Any]
_DATA_LOADERS: dict[str, DataLoaderFactory] = {
    "stage2_mixed_video": _stage2_loader,
    "stage3_r2r_natural": _stage3_r2r_loader,
    "stage3_r2r_balanced_single_action": _stage3_single_action_balanced_loader,
    "stage3_r2r_full_history_balanced_single_action": _stage3_single_action_full_history_loader,
}


def register_data_loader(name: str, factory: DataLoaderFactory, *, replace: bool = False) -> None:
    """Register a new data assembly without modifying current samplers."""

    if name in _DATA_LOADERS and not replace:
        raise KeyError(f"data loader already registered: {name}")
    _DATA_LOADERS[name] = factory


def build_data_loader(name: str, *, config: Any, **kwargs: Any) -> Any:
    """Build a registered loader/batch-builder from its unchanged config."""

    try:
        factory = _DATA_LOADERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown data loader {name!r}; available={sorted(_DATA_LOADERS)}") from exc
    return factory(config=config, **kwargs)


def available_data_loaders() -> tuple[str, ...]:
    return tuple(sorted(_DATA_LOADERS))
