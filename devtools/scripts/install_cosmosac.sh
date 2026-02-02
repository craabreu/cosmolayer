#!/usr/bin/env bash
set -euo pipefail

COSMOSAC_GIT_URL="${COSMOSAC_GIT_URL:-https://github.com/usnistgov/COSMOSAC}"
COSMOSAC_GIT_REF="${COSMOSAC_GIT_REF:-}"
COSMOSAC_WORKDIR="${COSMOSAC_WORKDIR:-${PWD}/.cosmosac}"
export COSMOSAC_WORKDIR

if [[ -d "${COSMOSAC_WORKDIR}" ]]; then
  rm -rf "${COSMOSAC_WORKDIR}"
fi

# Determine what ref to use (default to specific commit)
GIT_REF="${COSMOSAC_GIT_REF:-21dd92b}"

# Clone the repo
if ! git clone --recurse-submodules "${COSMOSAC_GIT_URL}" "${COSMOSAC_WORKDIR}"; then
  echo "Failed to clone ${COSMOSAC_GIT_URL}." >&2
  exit 1
fi

# Reset to the desired ref (works for commits, branches, and tags)
git -C "${COSMOSAC_WORKDIR}" reset --hard "${GIT_REF}"

# Ensure submodules are synced to the checked-out ref
if ! git -C "${COSMOSAC_WORKDIR}" submodule update --init --recursive --depth 1; then
  git -C "${COSMOSAC_WORKDIR}" submodule update --init --recursive
fi

# Apply patch to increase iteration count (post-clone, pre-build)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/cosmosac.patch"
if [[ -f "${PATCH_FILE}" ]]; then
  echo "Applying patch from ${PATCH_FILE}..."
  if git -C "${COSMOSAC_WORKDIR}" apply "${PATCH_FILE}"; then
    echo "Patch applied successfully"
  else
    echo "Warning: failed to apply patch (may already be applied or incompatible)" >&2
  fi
else
  echo "Warning: patch file not found at ${PATCH_FILE}; skipping patch" >&2
fi

# Check if COSMOSAC directory contains Python package metadata
INSTALL_DIR=""

# Check root directory first
if [[ -f "${COSMOSAC_WORKDIR}/pyproject.toml" ]]; then
  INSTALL_DIR="${COSMOSAC_WORKDIR}"
  echo "Found pyproject.toml at: ${INSTALL_DIR}"
elif [[ -f "${COSMOSAC_WORKDIR}/setup.py" ]]; then
  INSTALL_DIR="${COSMOSAC_WORKDIR}"
  echo "Found setup.py at: ${INSTALL_DIR}"
else
  # Search subdirectories for Python package metadata
  echo "Searching for Python package metadata in subdirectories..."
  PYPROJECT_PATH="$(find "${COSMOSAC_WORKDIR}" -name "pyproject.toml" -type f -print -quit 2>/dev/null || true)"
  SETUP_PATH="$(find "${COSMOSAC_WORKDIR}" -name "setup.py" -type f -print -quit 2>/dev/null || true)"
  
  if [[ -n "${PYPROJECT_PATH}" ]]; then
    INSTALL_DIR="$(dirname "${PYPROJECT_PATH}")"
    echo "Found pyproject.toml at: ${INSTALL_DIR}"
  elif [[ -n "${SETUP_PATH}" ]]; then
    INSTALL_DIR="$(dirname "${SETUP_PATH}")"
    echo "Found setup.py at: ${INSTALL_DIR}"
  fi
fi

if [[ -z "${INSTALL_DIR}" ]]; then
  echo "No Python packaging metadata found in ${COSMOSAC_WORKDIR}." >&2
  exit 1
fi

echo "Installing COSMOSAC package from: ${INSTALL_DIR}"
python -m pip install "${INSTALL_DIR}" --no-deps
