#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
.venv/bin/python -m pytest tests/test_sentiment.py -q --junitxml=verification/sentiment_gpu_host_pytest.xml
.venv/bin/python -m lab1.run --task sentiment --mode smoke --device cuda --output runs/part-b-cuda-smoke
.venv/bin/python scripts/benchmark_sentiment.py \
  --data-dir task2_sentiment/srinidhi/data_processed/full \
  --encoded-cache task2_sentiment/srinidhi/data_processed/full_encoded \
  --output verification/sentiment_5090_benchmark.json --steps 80 --workers 0 2
