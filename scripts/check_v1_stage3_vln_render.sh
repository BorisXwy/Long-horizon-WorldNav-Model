#!/usr/bin/env bash
set -euo pipefail

NAV_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
RUN_ROOT="${1:-$(ls -td /sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_* 2>/dev/null | head -1 || true)}"

if [[ -z "${RUN_ROOT}" || ! -d "${RUN_ROOT}" ]]; then
  echo "[vln-render] no rendered_obs run found" >&2
  exit 1
fi

echo "[vln-render] run_root=${RUN_ROOT}"
echo "[vln-render] size=$(du -sh "${RUN_ROOT}" 2>/dev/null | awk '{print $1}')"

SUMMARY="${RUN_ROOT}/manifests/render_summary.json"
if [[ -f "${SUMMARY}" ]]; then
  "${NAV_ROOT}/../virtual_env/.venv_infinite_world/bin/python" - <<PY
import json, pathlib
p = pathlib.Path("${SUMMARY}")
s = json.loads(p.read_text())
print(json.dumps({
    "selected_episodes": s.get("selected_episodes"),
    "completed": s.get("completed"),
    "errors": s.get("errors"),
    "frames": s.get("frames"),
    "selected_scenes": s.get("selected_scenes"),
    "per_source_selected": s.get("per_source_selected"),
    "sources": s.get("sources"),
}, ensure_ascii=False, indent=2))
PY
else
  echo "[vln-render] summary missing: ${SUMMARY}"
fi

echo "[vln-render] process:"
ps -u xiewenyuan -o pid,stat,etime,cmd | rg 'render_v1_stage3_vln_obs.py' || true

echo "[vln-render] gpu:"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
