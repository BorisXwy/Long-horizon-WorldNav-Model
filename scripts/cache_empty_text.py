#!/usr/bin/env python3
"""缓存 UMT5 对空字符串的真实 condition embedding。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "Infinite-World"))

from infworld.models.umt5 import T5EncoderModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = T5EncoderModel(
        model_max_length=512,
        dtype=torch.bfloat16,
        device=args.device,
        checkpoint_path=(
            "/sharedata/Wan2.1-T2V-1.3B/"
            "models_t5_umt5-xxl-enc-bf16.pth"
        ),
        tokenizer_path="/sharedata/Wan2.1-T2V-1.3B/google/umt5-xxl",
    )
    with torch.inference_mode():
        encoded = encoder.encode([""])
    torch.save(
        {
            "text": "",
            "y": encoded["y"].cpu().to(torch.bfloat16),
            "y_mask": encoded["y_mask"].cpu(),
        },
        args.output,
    )
    print(
        {
            "output": str(args.output),
            "y_shape": tuple(encoded["y"].shape),
            "active_tokens": int(encoded["y_mask"].sum()),
        }
    )


if __name__ == "__main__":
    main()
