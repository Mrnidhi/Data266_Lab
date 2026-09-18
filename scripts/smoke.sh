#!/usr/bin/env bash
# One-command reproduction of Srinidhi's synthetic CPU pipeline checks.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/prepare_python.sh
if [[ "$(uname -s)" == Linux ]]; then
  .venv/bin/python -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r requirements.txt -r requirements-image-metrics.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
.venv/bin/python scripts/rehearsal.py --mode smoke --device cpu --hourly-rate 0 --max-minutes 15 --output "reproducibility/raw_logs/srinidhi/smoke-$RUN_STAMP"
