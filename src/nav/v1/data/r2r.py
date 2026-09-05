"""R2R Stage3 policy data construction.

This module implements the current binding rule for VLN/R2R policy training:

- one training sample predicts exactly one future action chunk;
- ``Z_obs`` is a current observed T4 micro latent chunk;
- target action starts after the observed chunk, not at its first frame;
- history is a small number of previous micro chunks used to update Register;
- repeated terminal-observation padding chunks are not sampled as repeated
  ``STOP`` policy targets.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
import gzip
import json
import random
from pathlib import Path
from typing import Any

import torch

from nav.v1.data.stage2 import (
    ACTION_BASE,
    COMBO_DIM,
    FINAL_LATENT_H,
    FINAL_LATENT_T,
    FINAL_LATENT_W,
    FINAL_MICRO_FRAMES,
    FINAL_MICRO_STRIDE,
)


STOP_ACTION = 0
MOVE_FORWARD_ACTION = 1
TURN_LEFT_ACTION = 2
TURN_RIGHT_ACTION = 3
ACTION_NAMES = {
    STOP_ACTION: "STOP",
    MOVE_FORWARD_ACTION: "MOVE_FORWARD",
    TURN_LEFT_ACTION: "TURN_LEFT",
    TURN_RIGHT_ACTION: "TURN_RIGHT",
}

__all__ = [
    "ACTION_NAMES",
    "MOVE_FORWARD_ACTION",
    "R2RPolicyWindow",
    "R2RStage3DataConfig",
    "R2RStage3PolicyBatchBuilder",
    "STOP_ACTION",
    "TURN_LEFT_ACTION",
    "TURN_RIGHT_ACTION",
    "combo_to_name",
    "iter_jsonl",
    "open_text",
    "trans_rot_to_combo",
    "vln_action_to_combo",
]


@dataclass(slots=True)
class R2RStage3DataConfig:
    latent_manifest_dir: Path = Path(
        "/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/"
        "data/train/r2r_ce/t4_micro_latents_stoppad_20260822_1605/manifests"
    )
    rendered_manifest: Path = Path(
        "/sharedata/NAV/derived/v1/vln/rendered_obs/"
        "stage3_vln_render_r2r_train_stoppad_gpu0_20260822_1605/"
        "episodes/rendered_episodes.jsonl.gz"
    )
    text_empty: Path = Path("/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt")
    text_cache_root: Path = Path(
        "/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/"
        "data/train/r2r_ce/text_embeddings_stoppad_20260822_1605"
    )
    history_micro_choices: str = "1,2,3,4,5,6,7"
    action_horizon: int = 10
    batch_size: int = 1
    payload_cache_items: int = 2
    text_cache_items: int = 128
    require_text_cache: bool = True
    max_episodes: int = 0
    terminal_window_policy: str = "one_per_episode"
    action_oversample_mode: str = "none"
    action_label_alignment: str = "after_observation_chunk"
    turn_copy_bonus: int = 2
    stop_copy_bonus: int = 2
    max_copy_factor: int = 10
    seed: int = 20260823

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("latent_manifest_dir", "rendered_manifest", "text_empty", "text_cache_root"):
            payload[key] = str(payload[key])
        return payload


@dataclass(slots=True)
class R2RPolicyWindow:
    latent_path: Path
    sample_id: str
    dataset: str
    history_micro: int
    start_micro: int
    obs_micro: int
    label_start_action: int
    stop_index: int
    is_terminal: bool


def trans_rot_to_combo(trans_id: int, rot_id: int) -> int:
    trans_id = max(0, min(int(trans_id), ACTION_BASE - 1))
    rot_id = max(0, min(int(rot_id), ACTION_BASE - 1))
    return trans_id * ACTION_BASE + rot_id


def vln_action_to_combo(action_id: int) -> int:
    return {
        STOP_ACTION: trans_rot_to_combo(10, 0),
        MOVE_FORWARD_ACTION: trans_rot_to_combo(1, 0),
        TURN_LEFT_ACTION: trans_rot_to_combo(0, 3),
        TURN_RIGHT_ACTION: trans_rot_to_combo(0, 4),
    }.get(int(action_id), trans_rot_to_combo(10, 0))


def combo_to_name(combo_id: int) -> str:
    reverse = {vln_action_to_combo(key): value for key, value in ACTION_NAMES.items()}
    return reverse.get(int(combo_id), f"combo_{int(combo_id)}")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def iter_jsonl(path: Path):
    with open_text(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class R2RStage3PolicyBatchBuilder:
    """Windowed R2R policy sampler for full Stage3 training."""

    def __init__(self, cfg: R2RStage3DataConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.history_choices = [
            int(value.strip())
            for value in cfg.history_micro_choices.split(",")
            if value.strip()
        ]
        if not self.history_choices:
            raise ValueError("history_micro_choices must not be empty")
        if any(value < 0 for value in self.history_choices):
            raise ValueError("history_micro must be >= 0")
        if cfg.action_label_alignment not in {"after_observation_chunk", "reference_frame"}:
            raise ValueError(
                "action_label_alignment must be 'after_observation_chunk' or 'reference_frame', "
                f"got {cfg.action_label_alignment!r}"
            )
        if cfg.terminal_window_policy != "one_per_episode":
            raise ValueError("only terminal_window_policy='one_per_episode' is currently supported")
        if cfg.action_oversample_mode not in {"none", "copy_rare_actions"}:
            raise ValueError(f"unknown action_oversample_mode={cfg.action_oversample_mode!r}")
        if cfg.turn_copy_bonus < 0 or cfg.stop_copy_bonus < 0 or cfg.max_copy_factor < 1:
            raise ValueError("action copy bonuses must be non-negative and max_copy_factor must be >= 1")

        empty = torch.load(cfg.text_empty, map_location="cpu", weights_only=False)
        self.empty_y = empty["y"].to(torch.bfloat16)
        self.empty_mask = empty["y_mask"]
        self.payload_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.text_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, str]] = OrderedDict()
        self.text_hits = 0
        self.text_fallbacks = 0

        self.render_rows = self._load_render_rows(cfg.rendered_manifest)
        self.encoded_rows = self._load_encoded_rows(cfg.latent_manifest_dir)
        self.action_cache: dict[str, list[int]] = {}
        self.windows_by_k: dict[int, list[R2RPolicyWindow]] = {k: [] for k in self.history_choices}
        self.windows: list[R2RPolicyWindow] = []
        self._build_windows()
        if not self.windows:
            raise RuntimeError("no R2R Stage3 policy windows constructed")
        self.policy_combos = tuple(vln_action_to_combo(action_id) for action_id in ACTION_NAMES)
        self.raw_action_label_counts: Counter[int] = Counter()
        self.sampled_action_label_counts: Counter[int] = Counter()
        self.raw_action_label_counts_by_k: dict[int, Counter[int]] = {
            k: Counter() for k in self.history_choices
        }
        self.sampled_action_label_counts_by_k: dict[int, Counter[int]] = {
            k: Counter() for k in self.history_choices
        }
        self.copy_factor_counts: Counter[int] = Counter()
        self.window_copy_factors: dict[int, int] = {}
        self.sample_windows_by_k: dict[int, list[R2RPolicyWindow]] = {
            k: [] for k in self.history_choices
        }
        self._build_oversampled_windows()
        for values in self.sample_windows_by_k.values():
            self.rng.shuffle(values)
        self.cursors = {k: 0 for k in self.history_choices}

    @staticmethod
    def _load_render_rows(path: Path) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                rows[sample_id] = row
        if not rows:
            raise RuntimeError(f"empty rendered manifest: {path}")
        return rows

    @staticmethod
    def _load_encoded_rows(manifest_dir: Path) -> list[dict[str, Any]]:
        files = sorted(manifest_dir.glob("encoded_episodes_shard-*.jsonl"))
        if not files:
            files = sorted(manifest_dir.glob("*.jsonl"))
        rows: list[dict[str, Any]] = []
        for path in files:
            for row in iter_jsonl(path):
                if row.get("status") not in {None, "encoded"}:
                    continue
                latent_path = Path(str(row.get("latent_path", "")))
                if not latent_path.is_file():
                    continue
                rows.append(row)
        if not rows:
            raise RuntimeError(f"no encoded latent rows under {manifest_dir}")
        return rows

    @staticmethod
    def _read_actions(path: Path) -> list[int]:
        payload = json.loads(path.read_text())
        return [int(value) for value in payload.get("gt_actions", [])]

    @staticmethod
    def _first_stop(actions: list[int]) -> int | None:
        for index, action in enumerate(actions):
            if int(action) == STOP_ACTION:
                return index
        return None

    def _build_windows(self) -> None:
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
            for history_micro in self.history_choices:
                max_start = int(num_micro) - int(history_micro) - 1
                if max_start < 0:
                    continue
                for start_micro in range(max_start + 1):
                    obs_micro = start_micro + int(history_micro)
                    if self.cfg.action_label_alignment == "reference_frame":
                        label_start = obs_micro * FINAL_MICRO_STRIDE
                    else:
                        label_start = obs_micro * FINAL_MICRO_STRIDE + (FINAL_MICRO_FRAMES - 1)
                    if label_start > stop_index:
                        continue
                    is_terminal = label_start <= stop_index < label_start + self.cfg.action_horizon
                    window = R2RPolicyWindow(
                        latent_path=latent_path,
                        sample_id=sample_id,
                        dataset=str(render_row.get("dataset", "r2r_ce")),
                        history_micro=int(history_micro),
                        start_micro=int(start_micro),
                        obs_micro=int(obs_micro),
                        label_start_action=int(label_start),
                        stop_index=int(stop_index),
                        is_terminal=bool(is_terminal),
                    )
                    if is_terminal:
                        terminal_candidates.append(window)
                    else:
                        self._add_window(window)
            if terminal_candidates:
                # Keep exactly one terminal action chunk per episode.  Choose
                # the chunk whose label starts closest to STOP; tie-break by
                # longer Register history.
                chosen = max(
                    terminal_candidates,
                    key=lambda item: (item.label_start_action, item.history_micro),
                )
                self._add_window(chosen)

    def _add_window(self, window: R2RPolicyWindow) -> None:
        self.windows.append(window)
        self.windows_by_k[window.history_micro].append(window)

    def _build_oversampled_windows(self) -> None:
        """Build an in-memory copied window pool with ordinary CE semantics.

        The latent payload is never duplicated on disk.  Only references to a
        real window are repeated.  Windows with more TURN/STOP tokens receive
        more copies, capped to avoid a small set dominating the training run.
        """

        stop_combo = vln_action_to_combo(STOP_ACTION)
        turn_combos = {
            vln_action_to_combo(TURN_LEFT_ACTION),
            vln_action_to_combo(TURN_RIGHT_ACTION),
        }
        for window in self.windows:
            actions = self.action_cache[window.sample_id]
            target = self._action_window(actions, window.label_start_action)
            target_values = [int(value) for value in target.tolist()]
            present = set(target_values)
            unsupported = present.difference(self.policy_combos)
            if unsupported:
                raise RuntimeError(
                    f"R2R window produced unsupported action combos={sorted(unsupported)} "
                    f"sample={window.sample_id}"
                )
            self.raw_action_label_counts.update(target_values)
            self.raw_action_label_counts_by_k[window.history_micro].update(target_values)
            if self.cfg.action_oversample_mode == "none":
                copy_factor = 1
            else:
                turn_count = sum(value in turn_combos for value in target_values)
                stop_count = sum(value == stop_combo for value in target_values)
                copy_factor = min(
                    int(self.cfg.max_copy_factor),
                    1
                    + int(self.cfg.turn_copy_bonus) * turn_count
                    + int(self.cfg.stop_copy_bonus) * stop_count,
                )
            self.window_copy_factors[id(window)] = copy_factor
            self.copy_factor_counts[copy_factor] += 1
            self.sample_windows_by_k[window.history_micro].extend([window] * copy_factor)
            self.sampled_action_label_counts.update(
                {combo: count * copy_factor for combo, count in Counter(target_values).items()}
            )
            self.sampled_action_label_counts_by_k[window.history_micro].update(
                {combo: count * copy_factor for combo, count in Counter(target_values).items()}
            )

    def _history_uniform_action_ratio(
        self,
        counts_by_k: dict[int, Counter[int]],
    ) -> dict[int, float]:
        available = [counts for counts in counts_by_k.values() if sum(counts.values()) > 0]
        ratio = {combo: 0.0 for combo in self.policy_combos}
        for counts in available:
            total = float(sum(counts.values()))
            for combo in self.policy_combos:
                ratio[combo] += float(counts.get(combo, 0)) / total / float(len(available))
        return ratio

    def _payload(self, path: Path) -> dict[str, Any]:
        key = str(path)
        if key in self.payload_cache:
            payload = self.payload_cache.pop(key)
            self.payload_cache[key] = payload
            return payload
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.payload_cache[key] = payload
        while len(self.payload_cache) > max(0, int(self.cfg.payload_cache_items)):
            self.payload_cache.popitem(last=False)
        return payload

    def _actions_for(self, payload: dict[str, Any]) -> list[int]:
        sample_id = str(payload.get("sample_id", ""))
        if sample_id not in self.action_cache:
            path = Path(str(payload.get("action_path", "")))
            if not path.is_file():
                render_row = self.render_rows.get(sample_id, {})
                path = Path(str(render_row.get("episode_action_path", "")))
            self.action_cache[sample_id] = self._read_actions(path)
        return self.action_cache[sample_id]

    def _action_window(self, actions: list[int], start: int) -> torch.Tensor:
        stop_index = self._first_stop(actions)
        if stop_index is None:
            stop_index = len(actions) - 1
        out = []
        for index in range(start, start + self.cfg.action_horizon):
            if index < len(actions):
                out.append(vln_action_to_combo(actions[index]))
            elif index > stop_index:
                out.append(vln_action_to_combo(STOP_ACTION))
            else:
                out.append(vln_action_to_combo(STOP_ACTION))
        return torch.tensor(out, dtype=torch.long).clamp_(0, COMBO_DIM - 1)

    def _text(self, dataset: str, sample_id: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        key = f"{dataset}/{sample_id}"
        if key in self.text_cache:
            self.text_hits += 1
            value = self.text_cache.pop(key)
            self.text_cache[key] = value
            return value
        path = self.cfg.text_cache_root / dataset / f"{sample_id}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            value = (payload["y"].to(torch.bfloat16), payload["y_mask"], str(path))
            self.text_hits += 1
        else:
            if self.cfg.require_text_cache:
                raise FileNotFoundError(f"missing R2R instruction embedding: {path}")
            value = (self.empty_y, self.empty_mask, "empty_fallback_missing_r2r_instruction_embedding")
            self.text_fallbacks += 1
        self.text_cache[key] = value
        while len(self.text_cache) > max(0, int(self.cfg.text_cache_items)):
            self.text_cache.popitem(last=False)
        return value

    def _choose_k(self) -> int:
        available = [k for k, values in self.sample_windows_by_k.items() if values]
        return self.rng.choice(available)

    def _choose_window(self, history_micro: int) -> R2RPolicyWindow:
        values = self.sample_windows_by_k[history_micro]
        cursor = self.cursors[history_micro]
        if cursor >= len(values):
            self.rng.shuffle(values)
            cursor = 0
        self.cursors[history_micro] = cursor + 1
        return values[cursor]

    def next_batch(self, *, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        history_micro = self._choose_k()
        items = [self._choose_window(history_micro) for _ in range(self.cfg.batch_size)]
        histories: list[torch.Tensor] = []
        z_obs: list[torch.Tensor] = []
        a_hist: list[torch.Tensor] = []
        action_combo: list[torch.Tensor] = []
        action_masks: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        y_masks: list[torch.Tensor] = []
        sample_ids: list[str] = []
        text_sources: list[str] = []
        terminal_count = 0
        label_start_actions: list[int] = []

        for item in items:
            payload = self._payload(item.latent_path)
            chunks = payload["micro_latents"][0].float()
            expected = (16, FINAL_LATENT_T, FINAL_LATENT_H, FINAL_LATENT_W)
            if tuple(chunks.shape[1:]) != expected:
                raise RuntimeError(f"bad latent shape for {item.latent_path}: {tuple(chunks.shape)}")
            histories.append(chunks[item.start_micro : item.start_micro + item.history_micro])
            z_obs.append(chunks[item.obs_micro])
            actions = self._actions_for(payload)
            hist_rows = []
            for micro_index in range(item.start_micro, item.start_micro + item.history_micro):
                hist_start = micro_index * FINAL_MICRO_STRIDE + (FINAL_MICRO_FRAMES - 1)
                hist_rows.append(self._action_window(actions, hist_start))
            if hist_rows:
                a_hist.append(torch.stack(hist_rows, dim=0))
            else:
                a_hist.append(torch.empty(0, self.cfg.action_horizon, dtype=torch.long))
            target_combo = self._action_window(actions, item.label_start_action)
            action_combo.append(target_combo)
            action_masks.append(torch.ones(self.cfg.action_horizon, dtype=torch.float32))
            y, y_mask, source = self._text(item.dataset, item.sample_id)
            ys.append(y)
            y_masks.append(y_mask)
            text_sources.append(source)
            sample_ids.append(item.sample_id)
            terminal_count += int(item.is_terminal)
            label_start_actions.append(item.label_start_action)

        z_obs_t = torch.stack(z_obs, dim=0)
        y = torch.cat(ys, dim=0)
        y_mask = torch.cat(y_masks, dim=0)
        batch = {
            "history_latents": torch.stack(histories, dim=0).to(device=device, dtype=dtype, non_blocking=True),
            "z_obs": z_obs_t.to(device=device, dtype=dtype, non_blocking=True),
            "z_future_noisy": torch.zeros_like(z_obs_t).to(device=device, dtype=dtype, non_blocking=True),
            "visual_timestep": torch.zeros(self.cfg.batch_size, device=device, dtype=torch.float32),
            "a_hist_combo": torch.stack(a_hist, dim=0).to(device=device, non_blocking=True),
            "a_cur_combo": torch.zeros(self.cfg.batch_size, self.cfg.action_horizon, device=device, dtype=torch.long),
            "a_noise": torch.randn(self.cfg.batch_size, self.cfg.action_horizon, 6, device=device, dtype=dtype),
            "action_timestep": torch.rand(self.cfg.batch_size, device=device, dtype=torch.float32),
            "action_combo": torch.stack(action_combo, dim=0).to(device=device, non_blocking=True),
            "action_loss_mask": torch.stack(action_masks, dim=0).to(device=device, dtype=torch.float32, non_blocking=True),
            "action_target": torch.zeros(self.cfg.batch_size, self.cfg.action_horizon, 6, device=device, dtype=dtype),
            "y": y.to(device=device, dtype=dtype, non_blocking=True),
            "y_mask": y_mask.to(device=device, dtype=dtype, non_blocking=True),
        }
        meta = {
            "history_micro": history_micro,
            "sample_ids": sample_ids,
            "label_start_actions": label_start_actions,
            "copy_factors": [self.window_copy_factors[id(item)] for item in items],
            "terminal_count": terminal_count,
            "action_valid": float(batch["action_loss_mask"].sum().detach().cpu().item()),
            "text_hits": self.text_hits,
            "text_fallbacks": self.text_fallbacks,
            "text_sources": text_sources,
        }
        return batch, meta

    def summary(self) -> dict[str, Any]:
        terminal = 0
        by_k = {k: len(values) for k, values in self.windows_by_k.items()}
        sampled_by_k = {k: len(values) for k, values in self.sample_windows_by_k.items()}
        for window in self.windows:
            terminal += int(window.is_terminal)
        label_counts = self.raw_action_label_counts
        total_labels = sum(label_counts.values())
        natural_expected_ratio = self._history_uniform_action_ratio(self.raw_action_label_counts_by_k)
        oversampled_expected_ratio = self._history_uniform_action_ratio(self.sampled_action_label_counts_by_k)
        return {
            "episodes_encoded": len(self.encoded_rows),
            "episodes_rendered": len(self.render_rows),
            "windows": len(self.windows),
            "windows_by_history_micro": by_k,
            "terminal_windows": terminal,
            "nonterminal_windows": len(self.windows) - terminal,
            "history_micro_choices": self.history_choices,
            "action_label_alignment": self.cfg.action_label_alignment,
            "label_rule": (
                "label_start = obs_micro * FINAL_MICRO_STRIDE"
                if self.cfg.action_label_alignment == "reference_frame"
                else "label_start = obs_micro * FINAL_MICRO_STRIDE + (FINAL_MICRO_FRAMES - 1)"
            ),
            "target_semantics": (
                "one action chunk starting at the clean reference frame; repeated terminal padding chunks are not sampled"
                if self.cfg.action_label_alignment == "reference_frame"
                else "one future action chunk after Z_obs; repeated terminal padding chunks are not sampled"
            ),
            "action_oversample": {
                "mode": self.cfg.action_oversample_mode,
                "turn_copy_bonus": self.cfg.turn_copy_bonus,
                "stop_copy_bonus": self.cfg.stop_copy_bonus,
                "max_copy_factor": self.cfg.max_copy_factor,
                "virtual_windows": sum(sampled_by_k.values()),
                "virtual_windows_by_history_micro": sampled_by_k,
                "copy_factor_counts": {str(k): int(v) for k, v in sorted(self.copy_factor_counts.items())},
            },
            "action_label_counts": {combo_to_name(k): int(v) for k, v in sorted(label_counts.items())},
            "action_label_ratio": {
                combo_to_name(k): (float(v) / float(total_labels) if total_labels else 0.0)
                for k, v in sorted(label_counts.items())
            },
            "natural_history_uniform_action_ratio": {
                combo_to_name(k): v for k, v in natural_expected_ratio.items()
            },
            "oversampled_action_label_counts": {
                combo_to_name(k): int(v) for k, v in sorted(self.sampled_action_label_counts.items())
            },
            "oversampled_action_label_ratio": {
                combo_to_name(k): (
                    float(v) / float(sum(self.sampled_action_label_counts.values()))
                    if self.sampled_action_label_counts
                    else 0.0
                )
                for k, v in sorted(self.sampled_action_label_counts.items())
            },
            "oversampled_history_uniform_action_ratio": {
                combo_to_name(k): v for k, v in oversampled_expected_ratio.items()
            },
        }
