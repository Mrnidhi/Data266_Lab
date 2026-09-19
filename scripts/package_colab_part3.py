"""Build an allowlisted Part 3 source/data bundle with per-file checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def package(destination: Path):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    paths = set()
    for pattern in ('pyproject.toml', 'requirements*.txt', 'src/lab1/*.py',
                    'task3_gan/srinidhi/src/*.py', 'task3_gan/srinidhi/evaluate_local.py',
                    'task3_gan/srinidhi/config.json', 'task3_gan/srinidhi/data_processed/**/*.txt',
                    'task3_gan/srinidhi/data_processed/**/*.json',
                    'task3_gan/data/monet_jpg/*.jpg', 'task3_gan/data/photo_jpg/*.jpg',
                    'task3_gan/data/real_stats.npz',
                    'scripts/prepare_image_manifests.py', 'scripts/prepare_part3_data.py',
                    'scripts/prefetch_metric_weights.py', 'scripts/doctor.py',
                    'scripts/checkpoint_bundle.py', 'scripts/bootstrap_colab_part3.py',
                    'scripts/run_part3_colab.py',
                    'scripts/publish_cyclegan_results.py', 'tests/test_cyclegan.py',
                    'tests/test_image_manifests.py'):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    if len(list((ROOT / 'task3_gan/data/monet_jpg').glob('*.jpg'))) != 300:
        raise ValueError('Prepare verified real Part 3 images before packaging')
    metadata = {p.relative_to(ROOT).as_posix(): fingerprint(p) for p in sorted(paths)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('COLAB_INPUT_MANIFEST.json', json.dumps({'purpose': 'Part 3 training inputs; no credentials',
                         'files': metadata}, indent=2) + '\n')
    with zipfile.ZipFile(destination) as archive:
        for name, expected in metadata.items():
            content = archive.read(name)
            if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
                raise ValueError(f'Bundled input changed: {name}')
    with destination.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = {'archive': str(destination), 'bytes': destination.stat().st_size,
              'sha256': digest, 'files': len(metadata), 'verified': True}
    destination.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/part3-colab-inputs.zip')
    print(json.dumps(package(parser.parse_args().output), indent=2))
