#!/usr/bin/env bash
# One-command reproduction of Srinidhi's synthetic CPU pipeline checks.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/prepare_python.sh
if [[ "$(uname -s)" == Linux ]]; then
  .venv/bin/python -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r requirements.txt -r task3_gan/requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
for task in gpt sentiment cyclegan; do
  .venv/bin/python -m lab1.run --task "$task" --mode smoke --device cpu --output "runs/smoke-$RUN_STAMP/$task"
done
