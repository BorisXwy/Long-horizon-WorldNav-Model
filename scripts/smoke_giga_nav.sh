#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
OUT="${OUT:-$ROOT/log/giga_nav_smoke_$(date +%Y%m%d_%H%M%S)}"
python "$ROOT/scripts/train_giga_nav.py" --steps 1 --batch-size 1 --action-horizon 48 --device cuda:0 --output "$OUT"
python - "$OUT/config_and_structure.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
assert r["backbone_parameter_count"] > 1_000_000_000
assert "[B,16,4,56,112]" in r["obs_structure"]
print(json.dumps({"smoke":"full_giga_nav_forward_backward", "params":r["parameter_count"]}, ensure_ascii=False))
PY
