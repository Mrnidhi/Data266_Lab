#!/usr/bin/env bash
# Run inside an already authorized Linux GPU machine. Never rents or starts a pod.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null || { echo 'No NVIDIA GPU tools found; use the local CPU instructions.'; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
bash scripts/prepare_python.sh
.venv/bin/python -m pip install 'pip>=25'
.venv/bin/python -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt -r requirements-image-metrics.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python scripts/doctor.py --require-cuda --image-metrics --output verification/gpu_environment.json
echo 'Environment ready. No training has been started by this setup script.'
