#!/usr/bin/env bash
set -e -v
ruff format cosmolayer
ruff check --fix cosmolayer
ruff check --select I --fix cosmolayer
mypy cosmolayer
