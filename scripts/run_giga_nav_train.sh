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
# This environment was provisioned without the optional activate script in
# some installations.  Prefer activation when present, otherwise prepend its
# bin directory and invoke the interpreter explicitly; both paths use the
# same site-packages (including flash_attn).
if [[ -f "$ENV_ROOT/bin/activate" ]]; then
  source "$ENV_ROOT/bin/activate"
else
  export PATH="$ENV_ROOT/bin:$PATH"
fi
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
exec "$ENV_ROOT/bin/python" "$ROOT/scripts/train_giga_nav.py" "$@"
