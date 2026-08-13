#!/usr/bin/env bash
# Link shared VLN-CE / MP3D assets into StreamVLN and InternNav layouts (NAV-EVL-002).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
set -a
source "${NAV_ROOT}/config/r2r_ce_sota_reproduction.env"
set +a

link_dir() {
  local target="$1"
  local linkpath="$2"
  mkdir -p "$(dirname "${linkpath}")"
  if [[ -L "${linkpath}" ]]; then
    ln -sfn "${target}" "${linkpath}"
  elif [[ -e "${linkpath}" ]]; then
    echo "[skip] ${linkpath} exists and is not a symlink"
    return 0
  else
    ln -sfn "${target}" "${linkpath}"
  fi
  echo "[ok] ${linkpath} -> ${target}"
}

for p in "${R2R_VLNCE_ROOT}" "${RXR_VLNCE_ROOT}" "${MP3D_HABITAT_ROOT}"; do
  if [[ ! -d "${p}" ]]; then
    echo "[error] missing dataset path: ${p}" >&2
    exit 1
  fi
done

# StreamVLN expected layout
link_dir "${R2R_VLNCE_ROOT}" "${STREAMVLN_ROOT}/data/datasets/r2r"
link_dir "${RXR_VLNCE_ROOT}" "${STREAMVLN_ROOT}/data/datasets/rxr"
link_dir "${MP3D_HABITAT_ROOT}" "${STREAMVLN_ROOT}/data/scene_datasets/mp3d"
if [[ -f "${SCALEVLN_CE_JSON}" ]]; then
  mkdir -p "${STREAMVLN_ROOT}/data/datasets/scalevln"
  ln -sfn "${SCALEVLN_CE_JSON}" "${STREAMVLN_ROOT}/data/datasets/scalevln/scalevln_subset_150k.json.gz"
  echo "[ok] scalevln json linked"
fi

# InternNav: common data/ mirror + official Habitat eval layout
# (scripts/eval/configs/vln_r2r.yaml → data/vln_ce/raw_data/... & data/scene_data/mp3d_ce)
link_dir "${R2R_VLNCE_ROOT}" "${INTERNAV_ROOT}/data/datasets/r2r"
link_dir "${RXR_VLNCE_ROOT}" "${INTERNAV_ROOT}/data/datasets/rxr"
link_dir "${MP3D_HABITAT_ROOT}" "${INTERNAV_ROOT}/data/scene_datasets/mp3d"
link_dir "${R2R_VLNCE_ROOT}" "${INTERNAV_ROOT}/data/vln_ce/raw_data/r2r"
link_dir "${RXR_VLNCE_ROOT}" "${INTERNAV_ROOT}/data/vln_ce/raw_data/rxr"
# InternNav vln_r2r.yaml: scenes_dir=data/scene_data/mp3d_ce + episode scene_id=mp3d/<id>/...
# so link to parent of the mp3d folder (…/tasks), not …/tasks/mp3d.
link_dir "$(dirname "${MP3D_HABITAT_ROOT}")" "${INTERNAV_ROOT}/data/scene_data/mp3d_ce"

echo "Data links ready."
