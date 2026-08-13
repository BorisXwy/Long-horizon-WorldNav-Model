#!/usr/bin/env python3
"""验证 SpatialVID short latent 与 action sidecar 已配对。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latents",
        type=Path,
        default=Path("/sharedata/NAV/derived/latents/spatialvid"),
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=Path("/sharedata/NAV/derived/actions/spatialvid_short"),
    )
    parser.add_argument("--minimum", type=int, default=1000)
    args = parser.parse_args()

    latents = list(args.latents.glob("spatialvid__*.pt"))
    missing_actions = []
    bad_sidecars = []
    for path in latents:
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            # 历史完整 cache 允许没有 sidecar。
            continue
        try:
            metadata = json.loads(sidecar.read_text())
            cached = int(metadata["cached_chunks"])
            if cached < 2:
                bad_sidecars.append(path.stem)
            if (
                metadata.get("external_actions_required", False)
                and not (args.actions / f"{path.stem}.json").is_file()
            ):
                missing_actions.append(path.stem)
        except Exception:
            bad_sidecars.append(path.stem)
    report = {
        "latents": len(latents),
        "actions": len(list(args.actions.glob("spatialvid__*.json"))),
        "missing_actions": len(missing_actions),
        "bad_sidecars": len(bad_sidecars),
        "missing_examples": missing_actions[:10],
        "bad_examples": bad_sidecars[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (
        len(latents) < args.minimum
        or missing_actions
        or bad_sidecars
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
