#!/usr/bin/env python3
"""汇总三版 VBench 结果，并计算按视频配对的 bootstrap 置信区间。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


METRICS = (
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
LABELS = {
    "motion_smoothness": "Motion Smoothness",
    "dynamic_degree": "Dynamic Degree",
    "aesthetic_quality": "Aesthetic Quality",
    "imaging_quality": "Imaging Quality",
    "average": "四项平均",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infinite", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--expected-samples", type=int, default=10,
        help="三版应共同拥有的 prompt×seed 配对样本数",
    )
    return parser.parse_args()


def sample_key(path: str) -> str:
    name = Path(path).stem
    match = re.search(r"(\d{4}).*seed(\d{4})", name)
    if not match:
        raise ValueError(f"文件名缺少 prompt/seed：{name}")
    return f"prompt{match.group(1)}_seed{match.group(2)}"


def load_result(directory: Path) -> dict[str, dict[str, float]]:
    paths = sorted(directory.glob("*_eval_results.json"))
    if not paths:
        raise ValueError(f"{directory} 没有 eval_results.json")
    # 同一视频目录可能因断点续跑产生多次评测；使用时间戳最新结果。
    raw = json.loads(paths[-1].read_text())
    samples: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        for item in raw[metric][1]:
            key = sample_key(item["video_path"])
            value = float(item["video_results"])
            if metric == "imaging_quality" and value > 1:
                value /= 100.0
            samples.setdefault(key, {})[metric] = value
    for values in samples.values():
        values["average"] = float(np.mean([values[x] for x in METRICS]))
    return samples


def interval(values: np.ndarray, indices: np.ndarray) -> list[float]:
    means = values[indices].mean(axis=1)
    return [
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def main() -> None:
    args = parse_args()
    models = {
        "InfiniteWorld": load_result(args.infinite),
        "A": load_result(args.a),
        "B": load_result(args.b),
    }
    keys = sorted(set.intersection(*(set(x) for x in models.values())))
    if len(keys) != args.expected_samples:
        raise ValueError(
            f"预期 {args.expected_samples} 个配对样本，"
            f"实际 {len(keys)}：{keys}"
        )
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(keys), size=(args.bootstrap, len(keys)))
    metrics = (*METRICS, "average")
    summary: dict = {
        "num_paired_samples": len(keys),
        "sample_keys": keys,
        "bootstrap_repetitions": args.bootstrap,
        "models": {},
        "paired_delta_vs_infinite": {},
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for model_name, samples in models.items():
        arrays[model_name] = {
            metric: np.array([samples[key][metric] for key in keys])
            for metric in metrics
        }
        summary["models"][model_name] = {
            metric: interval(arrays[model_name][metric], indices)
            for metric in metrics
        }
    for model_name in ("A", "B"):
        summary["paired_delta_vs_infinite"][model_name] = {
            metric: interval(
                arrays[model_name][metric]
                - arrays["InfiniteWorld"][metric],
                indices,
            )
            for metric in metrics
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# 三版模型 VBench 统计子集结果",
        "",
        f"数值格式为均值 `[bootstrap 95% CI]`；共 "
        f"{args.expected_samples} 个 prompt×seed 配对样本。",
        "",
        "| 模型 | Motion Smoothness | Dynamic Degree | Aesthetic Quality | "
        "Imaging Quality | 四项平均 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in models:
        cells = []
        for metric in metrics:
            mean, low, high = summary["models"][model_name][metric]
            cells.append(f"{mean:.6f} [{low:.6f}, {high:.6f}]")
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 相对 InfiniteWorld 的配对差值",
        "",
        "| 模型 | Motion Smoothness | Dynamic Degree | Aesthetic Quality | "
        "Imaging Quality | 四项平均 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for model_name in ("A", "B"):
        cells = []
        for metric in metrics:
            mean, low, high = (
                summary["paired_delta_vs_infinite"][model_name][metric]
            )
            cells.append(f"{mean:+.6f} [{low:+.6f}, {high:+.6f}]")
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
