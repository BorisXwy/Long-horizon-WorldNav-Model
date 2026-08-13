#!/usr/bin/env python3
"""从既有 RE10K VGGT 结果构造可复现的相对置信度门槛。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=Path, default=Path("/sharedata/RealEstate10K/vggt_runs")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = []
    for path in sorted(args.runs.glob("*/pred_depth_conf.npy")):
        confidence = np.load(path, mmap_mode="r")
        summary = json.loads((path.parent / "summary.json").read_text())
        scenes.append(
            {
                "scene": path.parent.name,
                "confidence_p10": float(np.quantile(confidence, 0.10)),
                "confidence_p25": float(np.quantile(confidence, 0.25)),
                "confidence_p50": float(np.quantile(confidence, 0.50)),
                "confidence_p75": float(np.quantile(confidence, 0.75)),
                "camera_center_rmse": summary["camera_center_error"]["rmse"],
            }
        )
    if not scenes:
        raise RuntimeError(f"在 {args.runs} 中没有找到 RE10K VGGT 结果")

    p25 = np.array([row["confidence_p25"] for row in scenes])
    p50 = np.array([row["confidence_p50"] for row in scenes])
    payload = {
        "source": str(args.runs),
        "num_scenes": len(scenes),
        "confidence_semantics": "VGGT dense-depth confidence = 1 + exp(logit)",
        "reference_distribution": {
            "scene_confidence_p25": {
                f"q{q}": float(np.quantile(p25, q / 100))
                for q in (10, 25, 50, 75, 90)
            },
            "scene_confidence_p50": {
                f"q{q}": float(np.quantile(p50, q / 100))
                for q in (10, 25, 50, 75, 90)
            },
        },
        "quality_thresholds": {
            "reference_like": {
                "confidence_p25_min": float(np.quantile(p25, 0.25)),
                "confidence_p50_min": float(np.quantile(p50, 0.25)),
                "meaning": "不低于 RE10K 参考场景的下四分位水平",
            },
            "high_confidence": {
                "confidence_p25_min": float(np.quantile(p25, 0.50)),
                "confidence_p50_min": float(np.quantile(p50, 0.50)),
                "meaning": "达到 RE10K 参考场景的中位水平",
            },
        },
        "warning": (
            "该 confidence 是稠密深度置信度而非位姿正确概率；"
            "仅用于运动标签质量门控，不替代有真值的位姿误差评测。"
        ),
        "scenes": scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["quality_thresholds"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
