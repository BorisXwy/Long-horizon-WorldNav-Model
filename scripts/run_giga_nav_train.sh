#!/usr/bin/env bash
set -euo pipefail

# Persistent training launcher.  The Infinite-World virtualenv contains the
# CUDA-compatible flash_attn 2.7.4.post1 build; using this launcher avoids
# accidentally falling back to the system Python/SDPA path.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${NAV_VENV:-$ROOT/../virtual_env/.venv_infinite_world}"
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  echo "missing virtualenv Python: $ENV_ROOT/bin/python" >&2
  exit 2
fi
source "$ENV_ROOT/bin/activate"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
exec python "$ROOT/scripts/train_giga_nav.py" "$@"
