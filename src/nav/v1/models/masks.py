"""Attention masks for V1 action-centered DiT experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class V1TokenLayout:
    n_register: int
    n_obs: int
    n_future: int
    n_action: int

    @property
    def total(self) -> int:
        return self.n_register + self.n_obs + self.n_future + self.n_action

    @property
    def register(self) -> slice:
        return slice(0, self.n_register)

    @property
    def obs(self) -> slice:
        start = self.n_register
        return slice(start, start + self.n_obs)

    @property
    def future(self) -> slice:
        start = self.n_register + self.n_obs
        return slice(start, start + self.n_future)

    @property
    def action(self) -> slice:
        start = self.n_register + self.n_obs + self.n_future
        return slice(start, start + self.n_action)


def build_policy_safe_attention_mask(
    layout: V1TokenLayout,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return bool mask [query, key] where True means attention is allowed.

    Policy-safe constraints:
      - action tokens cannot attend to future visual tokens;
      - register tokens cannot attend to future visual tokens;
      - future tokens can attend to action/register/obs tokens.

    Text/timestep conditions are expected to be handled separately by the
    conditioner/cross-attention path and are not represented in this mask.
    """

    mask = torch.ones((layout.total, layout.total), dtype=torch.bool, device=device)
    if layout.n_future:
        mask[layout.action, layout.future] = False
        mask[layout.register, layout.future] = False
    return mask


def assert_policy_safe(mask: torch.Tensor, layout: V1TokenLayout) -> None:
    if layout.n_future == 0:
        return
    if torch.any(mask[layout.action, layout.future]):
        raise AssertionError("action/policy tokens can attend to future tokens")
    if torch.any(mask[layout.register, layout.future]):
        raise AssertionError("register tokens can attend to future tokens")

