"""Prepare an already allocated Colab runtime; never allocates or purchases compute.

Upload part3-colab-inputs.zip to /content first, then execute this file on Colab.
The active run stays on the VM's local disk. Download verified checkpoint backups
to the workstation between training segments, before releasing this runtime.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    archive = Path('/content/part3-colab-inputs.zip')
    destination = Path('/content/Data266_Lab')
    if destination.exists():
        raise FileExistsError('Existing project found; inspect it instead of overwriting a running experiment.')
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive entries')
        for name in names:
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError(f'Unsafe archive entry: {name}')
        manifest = json.loads(source.read('COLAB_INPUT_MANIFEST.json'))
        if set(names) != set(manifest['files']) | {'COLAB_INPUT_MANIFEST.json'}:
            raise ValueError('Input bundle inventory mismatch')
        for name, info in manifest['files'].items():
            content = source.read(name)
            if len(content) != info['bytes'] or hashlib.sha256(content).hexdigest() != info['sha256']:
                raise ValueError(f'Input checksum mismatch: {name}')
        source.extractall(destination)
    os.chdir(destination)
    subprocess.run(['nvidia-smi'], check=True)
    # Keep the Colab notebook environment intact by using a separate environment.
    if importlib.util.find_spec('ensurepip') is not None:
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
    else:
        # Colab's Python can omit ensurepip, which stdlib venv needs to seed pip.
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'virtualenv==20.34.0'], check=True)
        subprocess.run([sys.executable, '-m', 'virtualenv', '--no-download', '.venv'], check=True)
    python = str(destination / '.venv/bin/python')
    subprocess.run([python, '-m', 'pip', 'install', 'torch==2.11.0', 'torchvision==0.26.0',
                    '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
    subprocess.run([python, '-m', 'pip', 'install', '-r', 'requirements.txt',
                    '-r', 'requirements-image-metrics.txt'], check=True)
    subprocess.run([python, '-m', 'pip', 'install', '--no-deps', '-e', '.'], check=True)
    subprocess.run([python, '-m', 'pip', 'check'], check=True)
    subprocess.run([python, '-m', 'pytest', '-q', 'tests/test_cyclegan.py',
                    'tests/test_image_manifests.py'], check=True)
    subprocess.run([python, 'scripts/prefetch_metric_weights.py'], check=True)
    subprocess.run([python, 'scripts/doctor.py', '--require-cuda', '--image-metrics',
                    '--output', 'verification/colab_environment.json'], check=True)
    print('Part 3 dependencies, GPU tests, input hashes and metric weights verified.', flush=True)


if __name__ == '__main__':
    main()
