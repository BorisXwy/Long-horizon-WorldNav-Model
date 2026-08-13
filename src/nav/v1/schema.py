"""V1 shared schema for action-centered NAV samples.

This module intentionally stays light-weight: dataloaders may emit Python
dicts for speed, while these constants/helpers keep field semantics stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Literal


class Primitive(IntEnum):
    NOOP = 0
    STOP = 1
    MOVE_FORWARD = 2
    MOVE_BACKWARD = 3
    STRAFE_LEFT = 4
    STRAFE_RIGHT = 5
    TURN_LEFT = 6
    TURN_RIGHT = 7
    LOOK_UP = 8
    LOOK_DOWN = 9
    COMPOSITE = 10
    UNCERTAIN = 11


PRIMITIVE_NAMES = {item.value: item.name for item in Primitive}


ScaleType = Literal["metric", "simulator", "normalized", "pseudo", "unknown"]
ActionSource = Literal["video_pseudo", "vln_gt", "robot_executed", "pseudo_motion"]


@dataclass(slots=True)
class ActionStep:
    source: ActionSource
    primitive_id: int
    primitive_name: str
    delta_ego: list[float] = field(default_factory=lambda: [0.0] * 6)
    trans_magnitude: float = 0.0
    rot_magnitude: float = 0.0
    trans_bucket: int = 0
    rot_bucket: int = 0
    valid: bool = True
    is_stop: bool = False
    confidence: float = 1.0
    scale_type: ScaleType = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ActionChunk:
    steps: list[ActionStep]
    horizon: int
    valid_mask: list[bool]
    source: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload


def primitive_name(primitive_id: int) -> str:
    return PRIMITIVE_NAMES.get(int(primitive_id), "UNKNOWN")


def bucket_magnitude(value: float, *, small: float = 0.75, large: float = 2.0) -> int:
    """Bucket a normalized magnitude into 0=no/very small, 1=small, 2=medium, 3=large."""

    value = abs(float(value))
    if value < 1e-8:
        return 0
    if value < small:
        return 1
    if value < large:
        return 2
    return 3


def old_move_view_to_action_step(
    move: int,
    view: int,
    *,
    source: ActionSource = "video_pseudo",
    scale_type: ScaleType = "normalized",
    confidence: float = 1.0,
) -> ActionStep:
    """Map the V0 move/view labels to the V1 primitive table.

    V0 labels are coarse and sometimes diagonal. The first V1 implementation
    uses a dominant primitive plus keeps the raw move/view labels in metadata.
    """

    move = int(move)
    view = int(view)
    if move == 9 or view == 9:
        primitive = Primitive.UNCERTAIN
        confidence = min(confidence, 0.25)
    elif move in (1, 5, 6):
        primitive = Primitive.MOVE_FORWARD
    elif move in (2, 7, 8):
        primitive = Primitive.MOVE_BACKWARD
    elif move == 3:
        primitive = Primitive.STRAFE_LEFT
    elif move == 4:
        primitive = Primitive.STRAFE_RIGHT
    elif view in (3, 5, 7):
        primitive = Primitive.TURN_LEFT
    elif view in (4, 6, 8):
        primitive = Primitive.TURN_RIGHT
    elif view == 1:
        primitive = Primitive.LOOK_UP
    elif view == 2:
        primitive = Primitive.LOOK_DOWN
    else:
        primitive = Primitive.NOOP

    return ActionStep(
        source=source,
        primitive_id=int(primitive),
        primitive_name=primitive.name,
        confidence=confidence,
        scale_type=scale_type,
        is_stop=primitive is Primitive.STOP,
    )


def make_action_chunk_from_move_view(
    moves: list[int],
    views: list[int],
    *,
    horizon: int,
    source: ActionSource = "video_pseudo",
    scale_type: ScaleType = "normalized",
) -> ActionChunk:
    steps: list[ActionStep] = []
    valid_mask: list[bool] = []
    for index in range(horizon):
        if index < len(moves) and index < len(views):
            step = old_move_view_to_action_step(
                moves[index], views[index], source=source, scale_type=scale_type
            )
            valid = True
        else:
            step = ActionStep(
                source=source,
                primitive_id=int(Primitive.NOOP),
                primitive_name=Primitive.NOOP.name,
                valid=False,
                confidence=0.0,
                scale_type=scale_type,
            )
            valid = False
        steps.append(step)
        valid_mask.append(valid)

    primitive_counts: dict[int, int] = {}
    for step in steps:
        if step.valid:
            primitive_counts[step.primitive_id] = primitive_counts.get(step.primitive_id, 0) + 1
    dominant = max(primitive_counts, key=primitive_counts.get) if primitive_counts else int(Primitive.NOOP)
    confidence = float(sum(step.confidence for step in steps) / max(len(steps), 1))
    return ActionChunk(
        steps=steps,
        horizon=horizon,
        valid_mask=valid_mask,
        source=source,
        summary={
            "delta_total": [0.0] * 6,
            "dominant_primitive_id": int(dominant),
            "dominant_primitive_name": primitive_name(dominant),
            "confidence": confidence,
        },
    )

