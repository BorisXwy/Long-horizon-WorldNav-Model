#!/usr/bin/env python3
"""Mirror the lightweight GigaNav JSONL log into TensorBoard scalars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--logdir", type=Path, required=True)
    p.add_argument("--poll-seconds", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.logdir))
    last_step = -1
    try:
        while True:
            if args.jsonl.is_file():
                with args.jsonl.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        step = int(record.get("step", -1))
                        if step <= last_step:
                            continue
                        for key in ("loss", "grad_norm", "step_seconds"):
                            value = record.get(key)
                            if isinstance(value, (int, float)):
                                writer.add_scalar(f"train/{key}", value, step)
                        if "effective_batch_size" in record:
                            writer.add_scalar("train/effective_batch_size", record["effective_batch_size"], step)
                        last_step = step
                writer.flush()
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()


if __name__ == "__main__":
    main()
