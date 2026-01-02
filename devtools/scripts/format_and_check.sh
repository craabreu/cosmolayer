#!/usr/bin/env bash
set -e -v
ruff format cosmolayer
ruff check --fix cosmolayer
mypy cosmolayer