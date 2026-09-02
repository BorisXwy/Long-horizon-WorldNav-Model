#!/usr/bin/env bash
set -euo pipefail

# Compare the complete real-data GigaNav forward/backward path under the
# default interpreter and the project virtualenv (which has flash_attn).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
echo "system interpreter: $(python -c 'import torch; print(torch.__version__)')"
python - <<'PY'
try:
 import flash_attn; print("system flash_attn=available")
except Exception as exc: print("system flash_attn=unavailable", type(exc).__name__)
PY
echo "--- flash_attn virtualenv ---"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/../virtual_env/.venv_infinite_world/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
  PYTHON_BIN="$PYTHON_BIN" \
  OUT="${OUT:-$ROOT/log/giga_nav_attention_bench_$(date +%Y%m%d_%H%M%S)}" \
  bash "$ROOT/scripts/smoke_giga_nav.sh"
