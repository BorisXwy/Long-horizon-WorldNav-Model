"""R2R adapter for the GigaNav ablation.

The adapter intentionally reuses the canonical, real R2R latent/text/action
loader.  It only changes the model-facing view: a T4 observation chunk is
packed as Giga's reference-plus-future visual slots, while R2R actions are
converted to the four navigation classes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from nav.v1.data.r2r import (
    ACTION_NAMES,
    MOVE_FORWARD_ACTION,
    R2RStage3DataConfig,
    R2RStage3PolicyBatchBuilder,
    STOP_ACTION,
    TURN_LEFT_ACTION,
    TURN_RIGHT_ACTION,
)


@dataclass(slots=True)
class GigaNavDataConfig:
    latent_manifest_dir: Path = Path("/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/data/train/r2r_ce/t4_micro_latents_stoppad_20260822_1605/manifests")
    rendered_manifest: Path = Path("/sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_train_stoppad_gpu0_20260822_1605/episodes/rendered_episodes.jsonl.gz")
    text_empty: Path = Path("/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt")
    text_cache_root: Path = Path("/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/data/train/r2r_ce/text_embeddings_stoppad_20260822_1605")
    history_micro_choices: str = "1,2,3,4,5,6,7"
    action_horizon: int = 8
    batch_size: int = 1
    max_episodes: int = 0
    seed: int = 20260902

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("latent_manifest_dir", "rendered_manifest", "text_empty", "text_cache_root"):
            payload[key] = str(payload[key])
        return payload


class GigaNavR2RBatchBuilder:
    """Build complete model inputs from the existing R2R rendered dataset."""

    def __init__(self, cfg: GigaNavDataConfig) -> None:
        self.cfg = cfg
        self.base = R2RStage3PolicyBatchBuilder(
            R2RStage3DataConfig(
                latent_manifest_dir=cfg.latent_manifest_dir,
                rendered_manifest=cfg.rendered_manifest,
                text_empty=cfg.text_empty,
                text_cache_root=cfg.text_cache_root,
                history_micro_choices=cfg.history_micro_choices,
                action_horizon=cfg.action_horizon,
                batch_size=cfg.batch_size,
                max_episodes=cfg.max_episodes,
                action_oversample_mode="none",
                require_text_cache=True,
                seed=cfg.seed,
            )
        )

    @staticmethod
    def _combo_to_class(combo: torch.Tensor) -> torch.Tensor:
        mapping = {
            10 * 12 + 0: STOP_ACTION,
            1 * 12 + 0: MOVE_FORWARD_ACTION,
            0 * 12 + 3: TURN_LEFT_ACTION,
            0 * 12 + 4: TURN_RIGHT_ACTION,
        }
        output = torch.full_like(combo, STOP_ACTION)
        for key, value in mapping.items():
            output[combo == key] = value
        return output

    def next_batch(self, *, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        batch, meta = self.base.next_batch(device=device, dtype=dtype)
        combos = batch["action_combo"].long()
        batch = {
            "obs_latent": batch["z_obs"],
            "text_embedding": batch["y"],
            "text_mask": batch["y_mask"],
            "action_target": self._combo_to_class(combos),
            "action_loss_mask": batch["action_loss_mask"],
            "state": torch.zeros(batch["z_obs"].shape[0], 1, 14, device=device, dtype=dtype),
            "action_noise": torch.zeros(batch["z_obs"].shape[0], self.cfg.action_horizon, 14, device=device, dtype=dtype),
        }
        return batch, {**meta, "giga_action_target": "R2R combo mapped to STOP/MOVE_FORWARD/TURN_LEFT/TURN_RIGHT"}

    def summary(self) -> dict[str, Any]:
        return {"adapter": "GigaNavR2RBatchBuilder", "base": self.base.summary(), "config": self.cfg.to_dict(), "action_names": ACTION_NAMES}


class GigaNavAlignedWorldActionBatchBuilder(GigaNavR2RBatchBuilder):
    """Build causally aligned H=8 current/future/action samples.

    A cached T4 micro chunk represents 13 RGB frames.  We use latent plane 0
    as the current reference and planes 1:3 as the eight-transition visual
    target, so the action target starts at the same micro-chunk anchor instead
    of after the entire 12-transition chunk.
    """

    def next_batch(self, *, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        history_micro = self.base._choose_k()
        items = [self.base._choose_window(history_micro) for _ in range(self.cfg.batch_size)]
        references: list[torch.Tensor] = []
        futures: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        y_masks: list[torch.Tensor] = []
        sample_ids: list[str] = []
        label_starts: list[int] = []
        text_sources: list[str] = []
        for item in items:
            payload = self.base._payload(item.latent_path)
            chunks = payload["micro_latents"][0].float()
            obs = chunks[item.obs_micro]
            if tuple(obs.shape) != (16, 4, 56, 112):
                raise RuntimeError(f"bad T4 latent shape in {item.latent_path}: {tuple(obs.shape)}")
            references.append(obs[:, :1])
            futures.append(obs[:, 1:3])
            actions = self.base._actions_for(payload)
            label_start = int(item.obs_micro) * 12
            combos = self.base._action_window(actions, label_start)
            targets.append(self._combo_to_class(combos))
            masks.append(torch.ones(self.cfg.action_horizon, dtype=torch.float32))
            y, y_mask, source = self.base._text(item.dataset, item.sample_id)
            ys.append(y)
            y_masks.append(y_mask)
            sample_ids.append(item.sample_id)
            label_starts.append(label_start)
            text_sources.append(source)
        action_target = torch.stack(targets).to(device=device, non_blocking=True)
        batch = {
            "reference_latent": torch.stack(references).to(device=device, dtype=dtype, non_blocking=True),
            "future_latent": torch.stack(futures).to(device=device, dtype=dtype, non_blocking=True),
            "text_embedding": torch.cat(ys).to(device=device, dtype=dtype, non_blocking=True),
            "text_mask": torch.cat(y_masks).to(device=device, non_blocking=True),
            "action_target": action_target,
            "action_loss_mask": torch.stack(masks).to(device=device, dtype=torch.float32, non_blocking=True),
            "state": torch.zeros(self.cfg.batch_size, 1, 14, device=device, dtype=dtype),
        }
        meta = {
            "history_micro": history_micro,
            "sample_ids": sample_ids,
            "label_start_actions": label_starts,
            "text_sources": text_sources,
            "temporal_alignment": "Tref=1 + Tfuture=2 = 9 RGB frames / 8 transitions; action H=8",
        }
        return batch, meta

    def summary(self) -> dict[str, Any]:
        payload = super().summary()
        payload.update(
            {
                "adapter": "GigaNavAlignedWorldActionBatchBuilder",
                "temporal_alignment": "T4 cache sliced to latent[0:1] reference + latent[1:3] future",
                "action_alignment": "label_start = obs_micro * 12; H=8",
            }
        )
        return payload
