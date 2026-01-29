#!/usr/bin/env bash
set -euo pipefail

COSMOSAC_GIT_URL="${COSMOSAC_GIT_URL:-https://github.com/usnistgov/COSMOSAC}"
COSMOSAC_GIT_REF="${COSMOSAC_GIT_REF:-}"
COSMOSAC_WORKDIR="${COSMOSAC_WORKDIR:-${PWD}/.cosmosac}"
export COSMOSAC_WORKDIR

if [[ -d "${COSMOSAC_WORKDIR}" ]]; then
  rm -rf "${COSMOSAC_WORKDIR}"
fi

if ! git clone --depth 1 --recurse-submodules "${COSMOSAC_GIT_URL}" "${COSMOSAC_WORKDIR}"; then
  git clone --recurse-submodules "${COSMOSAC_GIT_URL}" "${COSMOSAC_WORKDIR}"
fi

if [[ -n "${COSMOSAC_GIT_REF}" ]]; then
  git -C "${COSMOSAC_WORKDIR}" fetch --depth 1 origin "${COSMOSAC_GIT_REF}"
  git -C "${COSMOSAC_WORKDIR}" checkout FETCH_HEAD
fi

if ! git -C "${COSMOSAC_WORKDIR}" submodule update --init --recursive --depth 1; then
  git -C "${COSMOSAC_WORKDIR}" submodule update --init --recursive
fi

INSTALL_DIR="$(python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["COSMOSAC_WORKDIR"])
candidates = []
for path in [root] + list(root.glob("**/pyproject.toml")) + list(
    root.glob("**/setup.py")
):
    if path.is_file():
        candidates.append(path.parent)

def score(path: Path) -> tuple[int, int]:
    depth = len(path.relative_to(root).parts)
    is_root = 0 if path == root else 1
    return (is_root, depth)

if candidates:
    best = sorted(set(candidates), key=score)[0]
    print(best)
PY
)"
INSTALL_DIR="$(echo "${INSTALL_DIR}" | head -n 1)"

if [[ -n "${INSTALL_DIR}" ]]; then
  echo "Found Python package at: ${INSTALL_DIR}"
  python -m pip install "${INSTALL_DIR}" --no-deps
else
  echo "No Python packaging metadata found in ${COSMOSAC_WORKDIR}." >&2
  exit 1
fi
