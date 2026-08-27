"""Full-prefix R2R sampling for long-horizon Register policy training.

Every policy target is paired with the complete latent/action prefix from the
start of its episode.  The prefix is not concatenated into Wan: it is consumed
recurrently by the fixed-size Register updater before the current ``Z_obs`` is
sent through the shared backbone.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from nav.v1.data.r2r import (
    ACTION_NAMES,
    R2RPolicyWindow,
    R2RStage3DataConfig,
    R2RStage3PolicyBatchBuilder,
    STOP_ACTION,
    vln_action_to_combo,
)
from nav.v1.data.sampler import COMBO_TO_ACTION_CLASS, SINGLE_ACTION_CLASSES
from nav.v1.data.stage2 import (
    FINAL_LATENT_H,
    FINAL_LATENT_T,
    FINAL_LATENT_W,
    FINAL_MICRO_FRAMES,
    FINAL_MICRO_STRIDE,
)


class FullHistoryBalancedSingleActionR2RBatchBuilder(R2RStage3PolicyBatchBuilder):
    """Balance four action classes while retaining each target's full prefix.

    Unlike the legacy window sampler, every constructed window satisfies
    ``start_micro == 0`` and ``history_micro == obs_micro``.  Variable prefix
    lengths are padded only for transport; ``history_lengths`` prevents padded
    chunks from entering the Register recurrence.
    """

    def __init__(
        self,
        cfg: R2RStage3DataConfig,
        *,
        history_action_horizon: int = 10,
    ) -> None:
        if cfg.action_horizon != 1:
            raise ValueError("full-history single-action builder requires action_horizon=1")
        if cfg.batch_size < 1:
            raise ValueError("batch_size must be positive")
        cfg.action_oversample_mode = "none"
        self.history_action_horizon = int(history_action_horizon)
        super().__init__(cfg)

        self.class_pools: dict[int, list[R2RPolicyWindow]] = {
            action_id: [] for action_id in SINGLE_ACTION_CLASSES
        }
        for window in self.windows:
            combo = int(self._action_window(self.action_cache[window.sample_id], window.label_start_action)[0])
            action_class = COMBO_TO_ACTION_CLASS.get(combo)
            if action_class is None:
                raise RuntimeError(f"unsupported single-action combo={combo} sample={window.sample_id}")
            self.class_pools[action_class].append(window)
        empty_classes = [ACTION_NAMES[key] for key, values in self.class_pools.items() if not values]
        if empty_classes:
            raise RuntimeError(f"full-history balanced sampler has empty classes: {empty_classes}")
        for values in self.class_pools.values():
            self.rng.shuffle(values)
        self.class_pool_cursors = {key: 0 for key in SINGLE_ACTION_CLASSES}
        self.class_cycle: list[int] = []
        self.class_cycle_cursor = 0
        self._reshuffle_class_cycle()

    def _build_windows(self) -> None:
        """Construct one full-prefix window for every valid current chunk."""

        selected_rows = self.encoded_rows[: self.cfg.max_episodes] if self.cfg.max_episodes > 0 else self.encoded_rows
        for row in selected_rows:
            sample_id = str(row["sample_id"])
            render_row = self.render_rows.get(sample_id)
            if render_row is None:
                continue
            action_path = Path(str(render_row.get("episode_action_path", "")))
            if not action_path.is_file():
                continue
            actions = self._read_actions(action_path)
            self.action_cache[sample_id] = actions
            stop_index = self._first_stop(actions)
            if stop_index is None:
                continue
            num_micro = int(row.get("num_micro_chunks") or render_row.get("v1_t4_num_micro_chunks") or 0)
            latent_path = Path(str(row["latent_path"]))
            terminal_candidates: list[R2RPolicyWindow] = []
            for obs_micro in range(1, num_micro):
                label_start = obs_micro * FINAL_MICRO_STRIDE + (FINAL_MICRO_FRAMES - 1)
                if label_start > stop_index:
                    continue
                is_terminal = label_start <= stop_index < label_start + self.cfg.action_horizon
                window = R2RPolicyWindow(
                    latent_path=latent_path,
                    sample_id=sample_id,
                    dataset=str(render_row.get("dataset", "r2r_ce")),
                    history_micro=obs_micro,
                    start_micro=0,
                    obs_micro=obs_micro,
                    label_start_action=label_start,
                    stop_index=stop_index,
                    is_terminal=bool(is_terminal),
                )
                if is_terminal:
                    terminal_candidates.append(window)
                else:
                    self.windows.append(window)
                    self.windows_by_k.setdefault(window.history_micro, []).append(window)
            if terminal_candidates:
                chosen = max(terminal_candidates, key=lambda item: item.label_start_action)
                self.windows.append(chosen)
                self.windows_by_k.setdefault(chosen.history_micro, []).append(chosen)

        self.windows_by_k = {key: values for key, values in self.windows_by_k.items() if values}
        self.history_choices = sorted(self.windows_by_k)

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
        stop_index = next(
            (index for index, value in enumerate(actions) if int(value) == STOP_ACTION),
            len(actions) - 1,
        )
        values = []
        for index in range(start, start + horizon):
            action_id = int(actions[index]) if index < len(actions) else STOP_ACTION
            if index > stop_index:
                action_id = STOP_ACTION
            values.append(vln_action_to_combo(action_id))
        return torch.tensor(values, dtype=torch.long)

    def next_batch(self, *, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        selected = [self._next_balanced_window() for _ in range(self.cfg.batch_size)]
        histories: list[torch.Tensor] = []
        history_actions: list[torch.Tensor] = []
        z_obs: list[torch.Tensor] = []
        target_combos: list[torch.Tensor] = []
        target_classes: list[int] = []
        ys: list[torch.Tensor] = []
        y_masks: list[torch.Tensor] = []
        text_sources: list[str] = []
        sample_ids: list[str] = []
        label_start_actions: list[int] = []
        terminal_count = 0

        for action_class, item in selected:
            payload = self._payload(item.latent_path)
            chunks = payload["micro_latents"][0].float()
            expected = (16, FINAL_LATENT_T, FINAL_LATENT_H, FINAL_LATENT_W)
            if tuple(chunks.shape[1:]) != expected:
                raise RuntimeError(f"bad latent shape for {item.latent_path}: {tuple(chunks.shape)}")
            if item.start_micro != 0 or item.history_micro != item.obs_micro:
                raise RuntimeError(f"full-prefix invariant violated: {item}")
            history = chunks[: item.history_micro]
            histories.append(history)
            z_obs.append(chunks[item.obs_micro])

            actions = self._actions_for(payload)
            hist_rows = []
            for micro_index in range(item.history_micro):
                hist_start = micro_index * FINAL_MICRO_STRIDE + (FINAL_MICRO_FRAMES - 1)
                hist_rows.append(self._combo_window(actions, hist_start, self.history_action_horizon))
            history_actions.append(torch.stack(hist_rows, dim=0))

            target_combo = self._combo_window(actions, item.label_start_action, 1)
            expected_combo = vln_action_to_combo(action_class)
            if int(target_combo[0]) != expected_combo:
                raise RuntimeError(
                    f"balanced class/target mismatch class={action_class} combo={int(target_combo[0])}"
                )
            target_combos.append(target_combo)
            target_classes.append(action_class)
            y, y_mask, text_source = self._text(item.dataset, item.sample_id)
            ys.append(y)
            y_masks.append(y_mask)
            text_sources.append(text_source)
            sample_ids.append(item.sample_id)
            label_start_actions.append(item.label_start_action)
            terminal_count += int(item.is_terminal)

        history_lengths = torch.tensor([history.shape[0] for history in histories], dtype=torch.long)
        history_batch = pad_sequence(histories, batch_first=True)
        history_action_batch = pad_sequence(history_actions, batch_first=True)
        z_obs_batch = torch.stack(z_obs, dim=0).to(device=device, dtype=dtype, non_blocking=True)
        batch_size = len(selected)
        batch = {
            "history_latents": history_batch.to(device=device, dtype=dtype, non_blocking=True),
            "history_lengths": history_lengths.to(device=device, non_blocking=True),
            "z_obs": z_obs_batch,
            "z_future_noisy": torch.zeros_like(z_obs_batch),
            "visual_timestep": torch.zeros(batch_size, device=device, dtype=torch.float32),
            "a_hist_combo": history_action_batch.to(device=device, non_blocking=True),
            "a_cur_combo": torch.zeros(batch_size, 1, device=device, dtype=torch.long),
            "a_noise": torch.zeros(batch_size, 1, 6, device=device, dtype=dtype),
            "action_timestep": torch.zeros(batch_size, device=device, dtype=torch.float32),
            "action_combo": torch.stack(target_combos, dim=0).to(device=device, non_blocking=True),
            "action_class": torch.tensor(target_classes, device=device, dtype=torch.long)[:, None],
            "action_loss_mask": torch.ones(batch_size, 1, device=device, dtype=torch.float32),
            "action_target": torch.zeros(batch_size, 1, 6, device=device, dtype=dtype),
            "y": torch.cat(ys, dim=0).to(device=device, dtype=dtype, non_blocking=True),
            "y_mask": torch.cat(y_masks, dim=0).to(device=device, dtype=dtype, non_blocking=True),
        }
        lengths = history_lengths.tolist()
        meta = {
            "history_micro": lengths,
            "history_micro_min": min(lengths),
            "history_micro_max": max(lengths),
            "history_micro_mean": float(sum(lengths)) / float(len(lengths)),
            "sample_ids": sample_ids,
            "label_start_actions": label_start_actions,
            "target_action_classes": target_classes,
            "target_action_names": [ACTION_NAMES[value] for value in target_classes],
            "terminal_count": terminal_count,
            "action_valid": float(batch_size),
            "text_hits": self.text_hits,
            "text_fallbacks": self.text_fallbacks,
            "text_sources": text_sources,
        }
        return batch, meta

    def summary(self) -> dict[str, Any]:
        payload = super().summary()
        natural_counts = Counter({ACTION_NAMES[key]: len(values) for key, values in self.class_pools.items()})
        total = sum(natural_counts.values())
        max_history = max(self.history_choices)
        payload["history_semantics"] = {
            "mode": "full_episode_prefix",
            "invariant": "start_micro=0 and history_micro=obs_micro for every target",
            "minimum_history_micro": min(self.history_choices),
            "maximum_history_micro": max_history,
            "maximum_history_frames": FINAL_MICRO_FRAMES + (max_history - 1) * FINAL_MICRO_STRIDE,
            "padding": "transport only; history_lengths controls per-sample Register rollout",
            "first_empty_history_step": "not sampled, matching the current Stage3 task",
        }
        payload["single_action"] = {
            "policy_horizon": 1,
            "history_action_horizon": self.history_action_horizon,
            "sampler": "balanced four-class cycle with replacement over full-prefix targets",
            "physical_batch_size": self.cfg.batch_size,
            "natural_window_counts": dict(natural_counts),
            "natural_window_ratio": {
                key: float(value) / float(total) for key, value in natural_counts.items()
            },
            "effective_sample_ratio": {ACTION_NAMES[key]: 0.25 for key in SINGLE_ACTION_CLASSES},
            "target_shape": ["B", 1],
        }
        return payload


class FullHistoryNaturalActionChunkR2RBatchBuilder(R2RStage3PolicyBatchBuilder):
    """Sample full-prefix windows in their natural distribution.

    This is the post-StreamVLN Stage3 policy protocol: every selected current
    observation predicts a four-action chunk, and windows are traversed once
    in shuffled natural order before reshuffling.  There is no class cycle,
    inverse-frequency weighting, or duplicated rare-action window.
    """

    def __init__(
        self,
        cfg: R2RStage3DataConfig,
        *,
        history_action_horizon: int = 10,
        model_action_horizon: int = 10,
    ) -> None:
        if cfg.action_horizon != 4:
            raise ValueError("full-history natural action-chunk builder requires action_horizon=4")
        if cfg.batch_size < 1:
            raise ValueError("batch_size must be positive")
        cfg.action_oversample_mode = "none"
        self.history_action_horizon = int(history_action_horizon)
        self.model_action_horizon = int(model_action_horizon)
        if self.model_action_horizon < cfg.action_horizon:
            raise ValueError("model action-token horizon must cover the supervised action chunk")
        super().__init__(cfg)
        self.natural_windows = list(self.windows)
        self.rng.shuffle(self.natural_windows)
        self.natural_cursor = 0

    def _build_windows(self) -> None:
        """Construct one full-prefix window for every valid current chunk."""

        selected_rows = self.encoded_rows[: self.cfg.max_episodes] if self.cfg.max_episodes > 0 else self.encoded_rows
        for row in selected_rows:
            sample_id = str(row["sample_id"])
            render_row = self.render_rows.get(sample_id)
            if render_row is None:
                continue
            action_path = Path(str(render_row.get("episode_action_path", "")))
            if not action_path.is_file():
                continue
            actions = self._read_actions(action_path)
            self.action_cache[sample_id] = actions
            stop_index = self._first_stop(actions)
            if stop_index is None:
                continue
            num_micro = int(row.get("num_micro_chunks") or render_row.get("v1_t4_num_micro_chunks") or 0)
            latent_path = Path(str(row["latent_path"]))
            terminal_candidates: list[R2RPolicyWindow] = []
            for obs_micro in range(1, num_micro):
                label_start = obs_micro * FINAL_MICRO_STRIDE + (FINAL_MICRO_FRAMES - 1)
                if label_start > stop_index:
                    continue
                is_terminal = label_start <= stop_index < label_start + self.cfg.action_horizon
                window = R2RPolicyWindow(
                    latent_path=latent_path,
                    sample_id=sample_id,
                    dataset=str(render_row.get("dataset", "r2r_ce")),
                    history_micro=obs_micro,
                    start_micro=0,
                    obs_micro=obs_micro,
                    label_start_action=label_start,
                    stop_index=stop_index,
                    is_terminal=bool(is_terminal),
                )
                if is_terminal:
                    terminal_candidates.append(window)
                else:
                    self.windows.append(window)
                    self.windows_by_k.setdefault(window.history_micro, []).append(window)
            if terminal_candidates:
                chosen = max(terminal_candidates, key=lambda item: item.label_start_action)
                self.windows.append(chosen)
                self.windows_by_k.setdefault(chosen.history_micro, []).append(chosen)

        self.windows_by_k = {key: values for key, values in self.windows_by_k.items() if values}
        self.history_choices = sorted(self.windows_by_k)

    def _next_natural_windows(self) -> list[R2RPolicyWindow]:
        selected = []
        for _ in range(self.cfg.batch_size):
            if self.natural_cursor >= len(self.natural_windows):
                self.rng.shuffle(self.natural_windows)
                self.natural_cursor = 0
            selected.append(self.natural_windows[self.natural_cursor])
            self.natural_cursor += 1
        return selected

    def next_batch(self, *, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        selected = self._next_natural_windows()
        histories: list[torch.Tensor] = []
        history_actions: list[torch.Tensor] = []
        z_obs: list[torch.Tensor] = []
        target_combos: list[torch.Tensor] = []
        target_classes: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        y_masks: list[torch.Tensor] = []
        text_sources: list[str] = []
        sample_ids: list[str] = []
        label_start_actions: list[int] = []
        terminal_count = 0

        for item in selected:
            payload = self._payload(item.latent_path)
            chunks = payload["micro_latents"][0].float()
            expected = (16, FINAL_LATENT_T, FINAL_LATENT_H, FINAL_LATENT_W)
            if tuple(chunks.shape[1:]) != expected:
                raise RuntimeError(f"bad latent shape for {item.latent_path}: {tuple(chunks.shape)}")
            if item.start_micro != 0 or item.history_micro != item.obs_micro:
                raise RuntimeError(f"full-prefix invariant violated: {item}")
            histories.append(chunks[: item.history_micro])
            z_obs.append(chunks[item.obs_micro])

            actions = self._actions_for(payload)
            hist_rows = []
            for micro_index in range(item.history_micro):
                hist_start = micro_index * FINAL_MICRO_STRIDE + (FINAL_MICRO_FRAMES - 1)
                hist_rows.append(self._action_window_with_horizon(actions, hist_start, self.history_action_horizon))
            history_actions.append(torch.stack(hist_rows, dim=0))

            target_combo = self._action_window(actions, item.label_start_action)
            action_classes = torch.tensor(
                [COMBO_TO_ACTION_CLASS.get(int(combo), -1) for combo in target_combo.tolist()],
                dtype=torch.long,
            )
            if bool((action_classes < 0).any()):
                raise RuntimeError(
                    f"unsupported action chunk combos={target_combo.tolist()} sample={item.sample_id}"
                )
            target_combos.append(target_combo)
            target_classes.append(action_classes)
            y, y_mask, text_source = self._text(item.dataset, item.sample_id)
            ys.append(y)
            y_masks.append(y_mask)
            text_sources.append(text_source)
            sample_ids.append(item.sample_id)
            label_start_actions.append(item.label_start_action)
            terminal_count += int(item.is_terminal)

        history_lengths = torch.tensor([history.shape[0] for history in histories], dtype=torch.long)
        history_batch = pad_sequence(histories, batch_first=True)
        history_action_batch = pad_sequence(history_actions, batch_first=True)
        z_obs_batch = torch.stack(z_obs, dim=0).to(device=device, dtype=dtype, non_blocking=True)
        target_class_batch = torch.stack(target_classes, dim=0)
        batch_size = len(selected)
        batch = {
            "history_latents": history_batch.to(device=device, dtype=dtype, non_blocking=True),
            "history_lengths": history_lengths.to(device=device, non_blocking=True),
            "z_obs": z_obs_batch,
            "z_future_noisy": torch.zeros_like(z_obs_batch),
            "visual_timestep": torch.zeros(batch_size, device=device, dtype=torch.float32),
            "a_hist_combo": history_action_batch.to(device=device, non_blocking=True),
            "a_cur_combo": torch.zeros(batch_size, self.model_action_horizon, device=device, dtype=torch.long),
            "a_noise": torch.zeros(batch_size, self.model_action_horizon, 6, device=device, dtype=dtype),
            "action_timestep": torch.zeros(batch_size, device=device, dtype=torch.float32),
            "action_combo": torch.stack(target_combos, dim=0).to(device=device, non_blocking=True),
            "action_class": target_class_batch.to(device=device, non_blocking=True),
            "action_loss_mask": torch.ones(
                batch_size,
                self.cfg.action_horizon,
                device=device,
                dtype=torch.float32,
            ),
            "action_target": torch.zeros(
                batch_size,
                self.cfg.action_horizon,
                6,
                device=device,
                dtype=dtype,
            ),
            "y": torch.cat(ys, dim=0).to(device=device, dtype=dtype, non_blocking=True),
            "y_mask": torch.cat(y_masks, dim=0).to(device=device, dtype=dtype, non_blocking=True),
        }
        lengths = history_lengths.tolist()
        class_rows = target_class_batch.tolist()
        meta = {
            "history_micro": lengths,
            "history_micro_min": min(lengths),
            "history_micro_max": max(lengths),
            "history_micro_mean": float(sum(lengths)) / float(len(lengths)),
            "sample_ids": sample_ids,
            "label_start_actions": label_start_actions,
            "target_action_classes": class_rows,
            "target_action_names": [
                [ACTION_NAMES[value] for value in row]
                for row in class_rows
            ],
            "terminal_count": terminal_count,
            "action_valid": float(batch_size * self.cfg.action_horizon),
            "text_hits": self.text_hits,
            "text_fallbacks": self.text_fallbacks,
            "text_sources": text_sources,
        }
        return batch, meta

    @staticmethod
    def _action_window_with_horizon(actions: list[int], start: int, horizon: int) -> torch.Tensor:
        stop_index = next(
            (index for index, value in enumerate(actions) if int(value) == STOP_ACTION),
            len(actions) - 1,
        )
        values = []
        for index in range(start, start + horizon):
            action_id = int(actions[index]) if index < len(actions) else STOP_ACTION
            if index > stop_index:
                action_id = STOP_ACTION
            values.append(vln_action_to_combo(action_id))
        return torch.tensor(values, dtype=torch.long)

    def summary(self) -> dict[str, Any]:
        payload = super().summary()
        max_history = max(self.history_choices)
        payload["history_semantics"] = {
            "mode": "full_episode_prefix",
            "invariant": "start_micro=0 and history_micro=obs_micro for every target",
            "minimum_history_micro": min(self.history_choices),
            "maximum_history_micro": max_history,
            "maximum_history_frames": FINAL_MICRO_FRAMES + (max_history - 1) * FINAL_MICRO_STRIDE,
            "padding": "transport only; history_lengths controls per-sample Register rollout",
            "first_empty_history_step": "not sampled, matching the current Stage3 task",
        }
        payload["action_chunk"] = {
            "policy_horizon": self.cfg.action_horizon,
            "backbone_action_tokens": self.model_action_horizon,
            "history_action_horizon": self.history_action_horizon,
            "sampler": "natural shuffled windows without replacement; no class balancing or duplication",
            "physical_batch_size": self.cfg.batch_size,
            "supervised_actions_per_batch": self.cfg.batch_size * self.cfg.action_horizon,
            "target_shape": ["B", self.cfg.action_horizon],
        }
        return payload


__all__ = [
    "FullHistoryBalancedSingleActionR2RBatchBuilder",
    "FullHistoryNaturalActionChunkR2RBatchBuilder",
]
