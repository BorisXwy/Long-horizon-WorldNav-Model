#!/usr/bin/env python3
"""Smoke-test V1 tokenization and policy-safe mask semantics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nav.v1.models.heads import ActionHead, FutureLatentHead
from nav.v1.models.masks import V1TokenLayout, assert_policy_safe, build_policy_safe_attention_mask
from nav.v1.models.stems import ActionStem, RegisterStem, VisualPatchStem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--register-dim", type=int, default=64)
    parser.add_argument("--register-tokens", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()

    b = 2
    z_obs = torch.randn(b, 16, 1, 56, 112)
    z_future = torch.randn(b, 16, 1, 56, 112)
    register = torch.randn(b, args.register_tokens, args.register_dim)

    visual_stem = VisualPatchStem(hidden_dim=args.hidden_dim)
    register_stem = RegisterStem(register_dim=args.register_dim, hidden_dim=args.hidden_dim)
    action_stem = ActionStem(hidden_dim=args.hidden_dim)
    action_head = ActionHead(hidden_dim=args.hidden_dim)
    future_head = FutureLatentHead(hidden_dim=args.hidden_dim)

    r_tokens = register_stem(register)
    obs_tokens = visual_stem(z_obs)
    future_tokens = visual_stem(z_future)
    action_tokens = action_stem.query(b, args.horizon, device=z_obs.device)

    layout = V1TokenLayout(
        n_register=r_tokens.shape[1],
        n_obs=obs_tokens.shape[1],
        n_future=future_tokens.shape[1],
        n_action=action_tokens.shape[1],
    )
    mask = build_policy_safe_attention_mask(layout)
    assert_policy_safe(mask, layout)

    action_out = action_head(action_tokens)
    future_out = future_head(future_tokens, latent_shape=(1, 56, 112))
    summary = {
        "register_tokens": list(r_tokens.shape),
        "obs_tokens": list(obs_tokens.shape),
        "future_tokens": list(future_tokens.shape),
        "action_tokens": list(action_tokens.shape),
        "mask_shape": list(mask.shape),
        "action_logits": list(action_out["primitive_logits"].shape),
        "future_out": list(future_out.shape),
        "policy_safe": True,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

