#!/usr/bin/env bash
# Prepare an isolated runtime without replacing system Python.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Use a Python 3.12 environment for this dependency set"'
  exit 0
fi
LAB1_PYTHON="${LAB1_PYTHON:-python3}"
if "$LAB1_PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3,12))'; then
  "$LAB1_PYTHON" -m venv .venv
else
  "$LAB1_PYTHON" -m venv .bootstrap
  .bootstrap/bin/python -m pip install 'uv>=0.8,<1'
  UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python" .bootstrap/bin/uv venv --python 3.12 --seed .venv
fi
