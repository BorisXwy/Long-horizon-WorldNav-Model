#!/usr/bin/env python3
"""Render V1 Stage3 VLN RGB observations with Habitat-Sim.

输入使用 `build_v1_stage3_vln_raw.py` 生成的 raw policy skeleton：

  /sharedata/NAV/derived/v1/vln/raw_policy/<run>/manifests/stage3_vln_episodes.jsonl.gz
  /sharedata/NAV/derived/v1/vln/raw_policy/<run>/actions/<dataset>/<variant>/<split>/ep*.json

输出只写 NAV 派生目录，不改 raw policy：

  /sharedata/NAV/derived/v1/vln/rendered_obs/<run_name>/
    frames/<dataset>/<variant>/<split>/ep<id>/*.png
    episodes/rendered_episodes.jsonl.gz
    manifests/render_summary.json

渲染语义：frame 00000 是 reset 后的当前 observation；随后按 GT action step，
frame t 对应执行 t 个 action 后的 observation。Stage3 policy chunk 的 obs_index=t
可直接映射到 `frames/.../{t:05d}.png`。
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    import habitat_sim
    from habitat_sim.agent import ActionSpec, ActuationSpec
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 habitat_sim；请使用 /mnt/pool1/sharehome/xiewenyuan/.conda/envs/unigoal/bin/python"
    ) from exc


DEFAULT_MP3D = Path("/sharedata/datasets/mp3d/v1/tasks/mp3d")
DEFAULT_RAW = Path("/sharedata/NAV/derived/v1/vln/raw_policy/20260810_030728")
DEFAULT_OUT = Path("/sharedata/NAV/derived/v1/vln/rendered_obs")

ACTION_ID_TO_NAME = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
}


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return path.open(mode.replace("t", ""))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def include_row(row: dict[str, Any], include: set[str]) -> bool:
    keys = {
        row["dataset"],
        f"{row['dataset']}:{row['variant']}",
        f"{row['dataset']}:{row['variant']}:{row['split']}",
    }
    return bool(keys & include)


def resolve_scene_path(scene_id: str, mp3d_root: Path) -> Path:
    path = Path(scene_id)
    name = path.name
    scan = path.parts[-2] if len(path.parts) >= 2 else name.replace(".glb", "")
    for candidate in [mp3d_root / scan / name, mp3d_root / scan / f"{scan}.glb"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到 MP3D scene: scene_id={scene_id} root={mp3d_root}")


def quat_habitat_from_xyzw(xyzw: list[float]):
    return np.quaternion(xyzw[3], xyzw[0], xyzw[1], xyzw[2])


def make_sim(
    scene_path: Path,
    *,
    height: int,
    width: int,
    hfov: float,
    sensor_height: float,
    forward_m: float,
    turn_deg: float,
    gpu: int,
) -> habitat_sim.Simulator:
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_path)
    backend.enable_physics = False
    backend.gpu_device_id = gpu

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "color_sensor"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [height, width]
    rgb.position = [0.0, sensor_height, 0.0]
    rgb.hfov = hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = sensor_height
    agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = [rgb]
    agent_cfg.action_space = {
        "stop": ActionSpec("stop"),
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=forward_m)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=turn_deg)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=turn_deg)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def obs_to_rgb(obs: dict[str, Any]) -> np.ndarray:
    rgb = np.asarray(obs["color_sensor"])
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb.astype(np.uint8, copy=False)


def load_action_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def episode_done(frame_dir: Path, expected_min_frames: int) -> bool:
    meta = frame_dir / "render_meta.json"
    if not meta.is_file():
        return False
    try:
        payload = json.loads(meta.read_text())
    except Exception:
        return False
    if int(payload.get("n_frames", 0)) < expected_min_frames:
        return False
    last = frame_dir / f"{expected_min_frames - 1:05d}.png"
    return last.is_file()


def write_frame(path: Path, rgb: np.ndarray, *, png_compress_level: int) -> None:
    tmp = path.with_suffix(".png.tmp")
    Image.fromarray(rgb).save(tmp, format="PNG", compress_level=png_compress_level)
    tmp.replace(path)


def render_episode(
    sim: habitat_sim.Simulator,
    row: dict[str, Any],
    action_payload: dict[str, Any],
    frame_dir: Path,
    *,
    max_steps: int,
    post_stop_pad_frames: int,
    post_stop_min_frames: int,
    png_compress_level: int,
) -> dict[str, Any]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    actions = [int(x) for x in action_payload.get("gt_actions", [])]
    if max_steps > 0:
        actions = actions[:max_steps]
    has_stop = any(ACTION_ID_TO_NAME.get(int(x)) == "stop" for x in actions)
    expected_min_frames = max(1, len(actions) + 1)
    if has_stop:
        expected_min_frames = max(
            expected_min_frames + max(0, int(post_stop_pad_frames)),
            max(1, int(post_stop_min_frames)),
        )
    if episode_done(frame_dir, expected_min_frames):
        payload = json.loads((frame_dir / "render_meta.json").read_text())
        payload["skipped_existing"] = True
        return payload

    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array(action_payload["start_position"], dtype=np.float32)
    state.rotation = quat_habitat_from_xyzw(action_payload["start_rotation"])
    agent.set_state(state)

    frame_paths = []
    render_times = []
    action_names = ["reset"]
    t0 = time.perf_counter()

    ts = time.perf_counter()
    obs = sim.get_sensor_observations()
    render_times.append(time.perf_counter() - ts)
    path0 = frame_dir / "00000.png"
    write_frame(path0, obs_to_rgb(obs), png_compress_level=png_compress_level)
    frame_paths.append(path0)

    for action_id in actions:
        name = ACTION_ID_TO_NAME.get(int(action_id))
        if name is None:
            raise ValueError(f"未知 VLN action_id={action_id} episode={row['episode_id']}")
        action_names.append(name)
        ts = time.perf_counter()
        if name == "stop":
            obs = sim.get_sensor_observations()
        else:
            obs = sim.step(name)
        render_times.append(time.perf_counter() - ts)
        path = frame_dir / f"{len(frame_paths):05d}.png"
        write_frame(path, obs_to_rgb(obs), png_compress_level=png_compress_level)
        frame_paths.append(path)
        if name == "stop":
            break

    if action_names[-1] == "stop":
        target_frames = max(
            len(frame_paths) + max(0, int(post_stop_pad_frames)),
            max(1, int(post_stop_min_frames)),
        )
        repeated_rgb = obs_to_rgb(obs)
        while len(frame_paths) < target_frames:
            path = frame_dir / f"{len(frame_paths):05d}.png"
            write_frame(path, repeated_rgb, png_compress_level=png_compress_level)
            frame_paths.append(path)
            action_names.append("stop_pad")

    wall = time.perf_counter() - t0
    meta = {
        "dataset": row["dataset"],
        "variant": row["variant"],
        "split": row["split"],
        "episode_id": row["episode_id"],
        "scene_id": row["scene_id"],
        "instruction": row.get("instruction", ""),
        "frame_dir": str(frame_dir),
        "n_actions": len(actions),
        "n_frames": len(frame_paths),
        "action_names": action_names,
        "post_stop_padding": {
            "enabled": action_names[-1] == "stop_pad",
            "pad_frames": max(0, int(post_stop_pad_frames)),
            "min_frames": max(1, int(post_stop_min_frames)),
            "rule": "after STOP, repeat the terminal observation so policy targets can become STOP STOP ...",
        },
        "timing_sec": {
            "wall": wall,
            "render_sum": float(np.sum(render_times)),
            "render_mean": float(np.mean(render_times)),
            "render_median": float(np.median(render_times)),
        },
        "skipped_existing": False,
    }
    (frame_dir / "render_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--mp3d-root", type=Path, default=DEFAULT_MP3D)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--run-name", default=f"stage3_vln_render_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--include", default="r2r_ce:standard:train,r2r_ce:standard:val_seen,r2r_ce:standard:val_unseen")
    ap.add_argument("--max-episodes-per-source", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument(
        "--post-stop-pad-frames",
        type=int,
        default=12,
        help="遇到 STOP 后额外重复 terminal observation 的帧数；用于构造 post-stop static chunk。",
    )
    ap.add_argument(
        "--post-stop-min-frames",
        type=int,
        default=97,
        help="含 STOP 的 episode 至少渲染到该帧数。默认 97 = 13 + 7*12，可支持 IW1 history + current obs。",
    )
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--sensor-height", type=float, default=1.25)
    ap.add_argument("--forward-m", type=float, default=0.25)
    ap.add_argument("--turn-deg", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--png-compress-level", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    include = {x.strip() for x in args.include.split(",") if x.strip()}
    episode_manifest = args.raw_root / "manifests" / "stage3_vln_episodes.jsonl.gz"
    if not episode_manifest.is_file():
        raise FileNotFoundError(episode_manifest)

    run_root = args.out_root / args.run_name
    frame_root = run_root / "frames"
    episode_out = run_root / "episodes" / "rendered_episodes.jsonl.gz"
    summary_path = run_root / "manifests" / "render_summary.json"
    episode_out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    per_source_seen: dict[str, int] = defaultdict(int)
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(episode_manifest):
        if not include_row(row, include):
            continue
        source_key = f"{row['dataset']}:{row['variant']}:{row['split']}"
        if args.max_episodes_per_source and per_source_seen[source_key] >= args.max_episodes_per_source:
            continue
        per_source_seen[source_key] += 1
        rows_by_scene[row["scene_id"]].append(row)

    stats: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw_root": str(args.raw_root),
        "run_root": str(run_root),
        "include": sorted(include),
        "height": args.height,
        "width": args.width,
        "gpu": args.gpu,
        "png_compress_level": args.png_compress_level,
        "selected_episodes": sum(len(v) for v in rows_by_scene.values()),
        "selected_scenes": len(rows_by_scene),
        "per_source_selected": dict(per_source_seen),
        "completed": 0,
        "skipped_existing": 0,
        "errors": 0,
        "frames": 0,
        "sources": defaultdict(lambda: {"episodes": 0, "frames": 0, "errors": 0}),
    }
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")

    with gzip.open(episode_out, "at") as out:
        for scene_index, (scene_id, rows) in enumerate(rows_by_scene.items(), 1):
            scene_path = resolve_scene_path(scene_id, args.mp3d_root)
            load_t0 = time.perf_counter()
            sim = make_sim(
                scene_path,
                height=args.height,
                width=args.width,
                hfov=args.hfov,
                sensor_height=args.sensor_height,
                forward_m=args.forward_m,
                turn_deg=args.turn_deg,
                gpu=args.gpu,
            )
            scene_load_sec = time.perf_counter() - load_t0
            print(json.dumps({
                "event": "scene_start",
                "scene_index": scene_index,
                "num_scenes": len(rows_by_scene),
                "scene_id": scene_id,
                "episodes": len(rows),
                "scene_load_sec": round(scene_load_sec, 3),
            }, ensure_ascii=False), flush=True)
            try:
                for row in rows:
                    source_key = f"{row['dataset']}:{row['variant']}:{row['split']}"
                    frame_dir = frame_root / row["dataset"] / row["variant"] / row["split"] / f"ep{row['episode_id']}"
                    try:
                        action_payload = load_action_payload(Path(row["episode_action_path"]))
                        meta = render_episode(
                            sim,
                            row,
                            action_payload,
                            frame_dir,
                            max_steps=args.max_steps,
                            post_stop_pad_frames=args.post_stop_pad_frames,
                            post_stop_min_frames=args.post_stop_min_frames,
                            png_compress_level=args.png_compress_level,
                        )
                        row_out = {
                            **row,
                            "render_status": "complete",
                            "render_frame_dir": str(frame_dir),
                            "render_num_frames": meta["n_frames"],
                            "render_meta_path": str(frame_dir / "render_meta.json"),
                            "render_run_root": str(run_root),
                        }
                        out.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                        out.flush()
                        stats["completed"] += 1
                        stats["frames"] += int(meta["n_frames"])
                        stats["sources"][source_key]["episodes"] += 1
                        stats["sources"][source_key]["frames"] += int(meta["n_frames"])
                        if meta.get("skipped_existing"):
                            stats["skipped_existing"] += 1
                    except Exception as exc:
                        stats["errors"] += 1
                        stats["sources"][source_key]["errors"] += 1
                        print(json.dumps({
                            "event": "episode_error",
                            "episode_id": row.get("episode_id"),
                            "source": source_key,
                            "error": repr(exc),
                        }, ensure_ascii=False), flush=True)
                    if stats["completed"] % max(1, args.log_every) == 0:
                        serializable_stats = {**stats, "sources": dict(stats["sources"])}
                        summary_path.write_text(json.dumps(serializable_stats, ensure_ascii=False, indent=2) + "\n")
                        print(json.dumps({
                            "event": "progress",
                            "completed": stats["completed"],
                            "selected": stats["selected_episodes"],
                            "frames": stats["frames"],
                            "errors": stats["errors"],
                        }, ensure_ascii=False), flush=True)
            finally:
                sim.close()
    serializable_stats = {**stats, "sources": dict(stats["sources"])}
    summary_path.write_text(json.dumps(serializable_stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "complete", **serializable_stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
