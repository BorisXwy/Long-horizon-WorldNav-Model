#!/usr/bin/env python3
"""Evaluate GigaNav by executing its first predicted action in Habitat."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
STREAMVLN_ROOT = PROJECT_ROOT / "StreamVLN"
DEFAULT_MODEL_PYTHON = PROJECT_ROOT / "virtual_env/.venv_infinite_world/bin/python"
sys.path.insert(0, str(STREAMVLN_ROOT / "streamvln"))

import habitat
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat_extensions import measures as _measures  # noqa: F401 - register VLN metrics
import numpy as np


def average(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def point_to_polyline_distance(position: list[float], reference_path: list[list[float]]) -> float:
    point = np.asarray([position[0], position[2]], dtype=np.float64)
    path = np.asarray([[value[0], value[2]] for value in reference_path], dtype=np.float64)
    if len(path) == 1:
        return float(np.linalg.norm(point - path[0]))
    best = float("inf")
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        scale = float(np.dot(segment, segment))
        alpha = 0.0 if scale == 0.0 else float(np.clip(np.dot(point - start, segment) / scale, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point - (start + alpha * segment))))
    return best


def local_route_direction(position: list[float], reference_path: list[list[float]]) -> np.ndarray | None:
    """Return the ordered GT segment direction nearest to an agent position."""

    point = np.asarray([position[0], position[2]], dtype=np.float64)
    path = np.asarray([[value[0], value[2]] for value in reference_path], dtype=np.float64)
    best_distance = float("inf")
    best_direction = None
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        scale = float(np.dot(segment, segment))
        if scale <= 1e-12:
            continue
        alpha = float(np.clip(np.dot(point - start, segment) / scale, 0.0, 1.0))
        distance = float(np.linalg.norm(point - (start + alpha * segment)))
        if distance < best_distance:
            best_distance = distance
            best_direction = segment / np.sqrt(scale)
    return best_direction


def phase_metrics(
    predicted_actions: list[int],
    target_actions: list[int],
    positions: list[list[float]],
    distances_to_goal: list[float],
    reference_path: list[list[float]],
    warmup_steps: int,
) -> dict[str, dict]:
    horizon = len(target_actions)
    aligned_start = min(warmup_steps, horizon)
    aligned_horizon = horizon - aligned_start
    boundaries = [
        aligned_start,
        aligned_start + aligned_horizon // 3,
        aligned_start + (2 * aligned_horizon) // 3,
        horizon,
    ]
    output = {}
    for name, start, end in zip(("early", "middle", "late"), boundaries[:-1], boundaries[1:]):
        available_end = min(end, len(predicted_actions))
        compared = max(0, available_end - start)
        correct = sum(
            int(predicted_actions[index] == target_actions[index])
            for index in range(start, available_end)
        )
        state_start = min(start, len(positions) - 1)
        state_end = min(end, len(positions) - 1)
        deviations = [
            point_to_polyline_distance(positions[index], reference_path)
            for index in range(state_start, state_end + 1)
        ]
        direction_cosines = []
        for index in range(state_start, state_end):
            displacement = np.asarray(
                [
                    positions[index + 1][0] - positions[index][0],
                    positions[index + 1][2] - positions[index][2],
                ],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(displacement))
            reference_direction = local_route_direction(positions[index], reference_path)
            if norm > 1e-5 and reference_direction is not None:
                direction_cosines.append(float(np.dot(displacement / norm, reference_direction)))
        output[name] = {
            "target_action_range": [start, end],
            "compared_actions": compared,
            "action_accuracy": float(correct) / max(compared, 1),
            "distance_to_goal_start": float(distances_to_goal[state_start]),
            "distance_to_goal_end": float(distances_to_goal[state_end]),
            "goal_progress": float(distances_to_goal[state_start] - distances_to_goal[state_end]),
            "mean_euclidean_deviation_to_reference_path": float(np.mean(deviations)),
            "movement_steps": len(direction_cosines),
            "mean_route_direction_cosine": (
                float(np.mean(direction_cosines)) if direction_cosines else 0.0
            ),
            "forward_route_direction_ratio": (
                float(np.mean(np.asarray(direction_cosines) > 0.0)) if direction_cosines else 0.0
            ),
        }
    return output


def save_trajectory_plot(
    output: Path,
    positions: list[list[float]],
    reference_path: list[list[float]],
    expert_horizon: int,
    warmup_steps: int,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = np.asarray(positions)
    reference = np.asarray(reference_path)
    aligned_start = min(warmup_steps, expert_horizon, len(predicted) - 1)
    aligned_horizon = max(0, expert_horizon - aligned_start)
    first = min(aligned_start + aligned_horizon // 3, len(predicted) - 1)
    second = min(aligned_start + (2 * aligned_horizon) // 3, len(predicted) - 1)
    third = min(expert_horizon, len(predicted) - 1)
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot(reference[:, 0], reference[:, 2], "k--o", linewidth=2, markersize=3, label="GT reference")
    segments = [
        (0, aligned_start, "tab:gray", "bootstrap"),
        (aligned_start, first, "tab:blue", "pred early"),
        (first, second, "tab:orange", "pred middle"),
        (second, third, "tab:green", "pred late"),
        (third, len(predicted) - 1, "tab:red", "pred after GT horizon"),
    ]
    for start, end, color, label in segments:
        if end > start:
            axis.plot(predicted[start : end + 1, 0], predicted[start : end + 1, 2], color=color, linewidth=2, label=label)
    axis.scatter(predicted[0, 0], predicted[0, 2], c="lime", edgecolors="black", s=70, label="start", zorder=5)
    axis.scatter(reference[-1, 0], reference[-1, 2], c="gold", edgecolors="black", s=90, marker="*", label="goal", zorder=5)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("Matterport x (m)")
    axis.set_ylabel("Matterport z (m)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def wait_for_socket(path: Path, process: subprocess.Popen, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError(f"inference server exited with code {process.returncode}")
        time.sleep(0.25)
    raise TimeoutError(f"inference server did not create {path} within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--model-python", type=Path, default=DEFAULT_MODEL_PYTHON)
    parser.add_argument("--habitat-config", type=Path, default=STREAMVLN_ROOT / "config/vln_r2r.yaml")
    parser.add_argument("--text-cache-root", type=Path, default=ROOT / "data/train/r2r_ce/text_embeddings_stoppad_20260822_1605/r2r_ce")
    parser.add_argument(
        "--ground-truth-action-root",
        type=Path,
        default=Path("/sharedata/NAV/derived/v1/vln/raw_policy/20260810_030728/actions/r2r_ce/standard"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", type=Path, default=Path("/tmp/nav_giga_closed_loop.sock"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    result_jsonl = args.output / "episodes.jsonl"
    summary_path = args.output / "summary.json"
    server_log_path = args.output / "inference_server.log"
    args.socket.unlink(missing_ok=True)

    server_command = [
        str(args.model_python),
        str(ROOT / "scripts/giga_nav_closed_loop_server.py"),
        "--checkpoint", str(args.checkpoint),
        "--socket", str(args.socket),
        "--device", args.device,
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    server_log = server_log_path.open("w")
    server = subprocess.Popen(server_command, stdout=server_log, stderr=subprocess.STDOUT, env=environment)
    connection = None
    env = None
    rows: list[dict] = []
    action_counts: Counter[int] = Counter()
    try:
        wait_for_socket(args.socket, server)
        connection = Client(str(args.socket), family="AF_UNIX")
        ready = connection.recv()
        if ready.get("status") != "ready":
            raise RuntimeError(f"bad server handshake: {ready}")
        print(json.dumps({"event": "server_ready", **ready}, ensure_ascii=False), flush=True)
        reference_delay = int(ready["training_reference_delay_steps"])
        window_frames = int(ready["online_vae_input_frames"])
        encoded_frames = int(ready["online_vae_encoded_frames"])

        config = get_habitat_config(str(args.habitat_config))
        with habitat.config.read_write(config):
            config.habitat.dataset.split = args.split
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_device_id
            config.habitat.environment.iterator_options.shuffle = False
        env = habitat.Env(config=config)
        episodes = list(env.episodes)
        if args.max_episodes > 0:
            episodes = episodes[: args.max_episodes]
        if not episodes:
            raise RuntimeError(f"no Habitat episodes for split={args.split}")

        for episode_index, episode in enumerate(episodes, 1):
            episode_id = str(episode.episode_id)
            sample_id = f"r2r_ce__standard__{args.split}__ep{episode_id}"
            text_path = args.text_cache_root / f"{sample_id}.pt"
            if not text_path.is_file():
                raise FileNotFoundError(f"missing instruction embedding: {text_path}")
            ground_truth_path = args.ground_truth_action_root / args.split / f"ep{episode_id}.json"
            if not ground_truth_path.is_file():
                raise FileNotFoundError(f"missing ground-truth action trace: {ground_truth_path}")
            ground_truth = json.loads(ground_truth_path.read_text())
            target_actions = [int(value) for value in ground_truth["gt_actions"]]
            reference_path = [[float(value) for value in point] for point in ground_truth["reference_path"]]
            env.current_episode = episode
            observation = env.reset()
            connection.send({"command": "reset", "episode_id": episode_id, "text_embedding": str(text_path)})
            reset_reply = connection.recv()
            if reset_reply.get("status") != "reset":
                raise RuntimeError(f"server reset failed: {reset_reply}")

            steps = 0
            episode_actions: Counter[int] = Counter()
            latency_rows: list[dict] = []
            cache_hits = 0
            rgb_history: deque[np.ndarray] = deque(maxlen=window_frames)
            predicted_actions: list[int] = []
            positions = [env.sim.get_agent_state().position.astype(float).tolist()]
            distances_to_goal = [float(env.get_metrics()["distance_to_goal"])]
            while not env.episode_over:
                current_rgb = np.asarray(observation["rgb"], dtype=np.uint8).copy()
                rgb_history.append(current_rgb)
                padded_window = [rgb_history[0]] * (window_frames - len(rgb_history)) + list(rgb_history)
                connection.send({"command": "infer", "rgb_window": np.stack(padded_window, axis=0)})
                reply = connection.recv()
                if reply.get("status") != "ok":
                    raise RuntimeError(f"inference failed: {reply}")
                action = int(reply["action"])
                action_counts[action] += 1
                episode_actions[action] += 1
                predicted_actions.append(action)
                cache_hits += int(reply.get("cache_hit", False))
                latency_rows.append({
                    key: float(reply[key])
                    for key in ("preprocess_seconds", "vae_seconds", "policy_seconds", "total_seconds")
                })
                observation = env.step(action)
                steps += 1
                positions.append(env.sim.get_agent_state().position.astype(float).tolist())
                distances_to_goal.append(float(env.get_metrics()["distance_to_goal"]))
                if steps <= 3 or steps % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "step",
                                "episode_id": episode_id,
                                "step": steps,
                                "action": action,
                                "action_sequence": reply["action_sequence"],
                                "latency": latency_rows[-1],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

            metrics = env.get_metrics()
            phases = phase_metrics(
                predicted_actions,
                target_actions,
                positions,
                distances_to_goal,
                reference_path,
                reference_delay,
            )
            trajectory_path = args.output / "trajectories" / f"ep{episode_id}.json"
            trajectory_plot = args.output / "trajectories" / f"ep{episode_id}.png"
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "instruction": episode.instruction.instruction_text,
                        "predicted_actions": predicted_actions,
                        "target_actions": target_actions,
                        "positions": positions,
                        "distances_to_goal": distances_to_goal,
                        "reference_path": reference_path,
                        "phase_metrics": phases,
                        "training_reference_delay_steps": reference_delay,
                        "bootstrap_action_range": [0, min(reference_delay, len(target_actions))],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            save_trajectory_plot(
                trajectory_plot,
                positions,
                reference_path,
                len(target_actions),
                reference_delay,
                f"R2R {args.split} ep{episode_id}",
            )
            row = {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "scene_id": episode.scene_id,
                "instruction": episode.instruction.instruction_text,
                "steps": steps,
                "success": float(metrics["success"]),
                "spl": float(metrics["spl"]),
                "oracle_success": float(metrics["oracle_success"]),
                "distance_to_goal": float(metrics["distance_to_goal"]),
                "action_counts": {str(key): int(value) for key, value in sorted(episode_actions.items())},
                "inference_cache_hits": cache_hits,
                "expert_action_horizon": len(target_actions),
                "training_reference_delay_steps": reference_delay,
                "bootstrap_action_range": [0, min(reference_delay, len(target_actions))],
                "phase_metrics": phases,
                "trajectory_path": str(trajectory_path),
                "trajectory_plot": str(trajectory_plot),
                "latency": {
                    key: average(latency_rows, key)
                    for key in ("preprocess_seconds", "vae_seconds", "policy_seconds", "total_seconds")
                },
            }
            rows.append(row)
            with result_jsonl.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"event": "episode", **row}, ensure_ascii=False), flush=True)

        total_steps = sum(int(row["steps"]) for row in rows)
        total_cache_hits = sum(int(row["inference_cache_hits"]) for row in rows)
        uncached_inferences = total_steps - total_cache_hits
        latency_totals = {
            key: sum(float(row["latency"][key]) * int(row["steps"]) for row in rows)
            for key in ("preprocess_seconds", "vae_seconds", "policy_seconds", "total_seconds")
        }
        action_total = sum(action_counts.values())
        aggregate_phases = {
            phase: {
                metric: float(np.mean([row["phase_metrics"][phase][metric] for row in rows]))
                for metric in (
                    "action_accuracy",
                    "goal_progress",
                    "mean_euclidean_deviation_to_reference_path",
                    "mean_route_direction_cosine",
                    "forward_route_direction_ratio",
                )
            }
            for phase in ("early", "middle", "late")
        }
        summary = {
            "method": "GigaNav Wan2.1-1.3B H8",
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "episodes": len(rows),
            "protocol": (
                f"online Habitat; checkpoint-derived rolling RGB window T={window_frames}; "
                f"training reference delay={reference_delay}; Wan causal VAE -> T4; "
                f"encode effective reference frames={encoded_frames}; "
                "predict H8; execute first action; replan every environment step"
            ),
            "training_reference_delay_steps": reference_delay,
            "bootstrap_semantics": (
                "before 13 observations exist, left-pad the rolling window with the episode's first RGB; "
                "phase metrics exclude these bootstrap steps"
                if reference_delay else "none"
            ),
            "success_rate": average(rows, "success"),
            "spl": average(rows, "spl"),
            "oracle_success": average(rows, "oracle_success"),
            "navigation_error": average(rows, "distance_to_goal"),
            "mean_steps": average(rows, "steps"),
            "phase_metrics": aggregate_phases,
            "action_counts": {str(key): int(value) for key, value in sorted(action_counts.items())},
            "action_ratio": {
                str(key): float(value) / max(action_total, 1)
                for key, value in sorted(action_counts.items())
            },
            "inference_cache": {
                "hits": total_cache_hits,
                "uncached": uncached_inferences,
                "hit_ratio": float(total_cache_hits) / max(total_steps, 1),
                "semantics": "exact reuse for identical effective reference RGB within one episode; valid because GigaNav has no recurrent state",
            },
            "mean_latency_per_environment_step": {
                key: value / max(total_steps, 1) for key, value in latency_totals.items()
            },
            "mean_fresh_inference_components": {
                key: latency_totals[key] / max(uncached_inferences, 1)
                for key in ("preprocess_seconds", "vae_seconds", "policy_seconds")
            },
            "action_ids": {"0": "STOP", "1": "MOVE_FORWARD", "2": "TURN_LEFT", "3": "TURN_RIGHT"},
            "episodes_path": str(result_jsonl),
            "server_log": str(server_log_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"event": "summary", **summary}, ensure_ascii=False), flush=True)
    finally:
        if connection is not None:
            try:
                connection.send({"command": "close"})
                connection.recv()
            except Exception:
                pass
            connection.close()
        if env is not None:
            env.close()
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        server_log.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
