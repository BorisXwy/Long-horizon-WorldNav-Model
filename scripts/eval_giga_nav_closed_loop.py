#!/usr/bin/env python3
"""Evaluate GigaNav by executing its first predicted action in Habitat."""

from __future__ import annotations

import argparse
from collections import Counter
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
            while not env.episode_over:
                connection.send({"command": "infer", "rgb": np.asarray(observation["rgb"], dtype=np.uint8)})
                reply = connection.recv()
                if reply.get("status") != "ok":
                    raise RuntimeError(f"inference failed: {reply}")
                action = int(reply["action"])
                action_counts[action] += 1
                episode_actions[action] += 1
                cache_hits += int(reply.get("cache_hit", False))
                latency_rows.append({
                    key: float(reply[key])
                    for key in ("preprocess_seconds", "vae_seconds", "policy_seconds", "total_seconds")
                })
                observation = env.step(action)
                steps += 1
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
        summary = {
            "method": "GigaNav Wan2.1-1.3B H8",
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "episodes": len(rows),
            "protocol": "online Habitat; current RGB -> causal Wan VAE T1 reference, pad three zero future planes to T4; predict H8; execute first action; replan every environment step",
            "success_rate": average(rows, "success"),
            "spl": average(rows, "spl"),
            "oracle_success": average(rows, "oracle_success"),
            "navigation_error": average(rows, "distance_to_goal"),
            "mean_steps": average(rows, "steps"),
            "action_counts": {str(key): int(value) for key, value in sorted(action_counts.items())},
            "action_ratio": {
                str(key): float(value) / max(action_total, 1)
                for key, value in sorted(action_counts.items())
            },
            "inference_cache": {
                "hits": total_cache_hits,
                "uncached": uncached_inferences,
                "hit_ratio": float(total_cache_hits) / max(total_steps, 1),
                "semantics": "exact reuse for identical RGB within one episode; valid because GigaNav has no recurrent state",
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
