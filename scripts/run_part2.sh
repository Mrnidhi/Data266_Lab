#!/usr/bin/env bash
# Run on the authorized GPU machine after bootstrap and benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
lab_output=${1:-runs/part-b-full}
lab_workers=${2:-0}
export PYTHONUNBUFFERED=1
exec .venv/bin/python -m lab1.run \
  --task sentiment --mode full --device cuda \
  --set 'data_dir="task2_sentiment/srinidhi/data_processed/full"' \
  --set 'encoded_cache="task2_sentiment/srinidhi/data_processed/full_encoded"' \
  --set "num_workers=$lab_workers" \
  --set 'log_every_steps=50' \
  --output "$lab_output"
