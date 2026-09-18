"""Create a portable preparation bundle; never label it a final submission."""
from __future__ import annotations
import hashlib
import argparse
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metric-weights", action="store_true", help="Also create a separate optional frozen-evaluation-weight archive")
args = parser.parse_args()
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)
paths = set()
for pattern in (
    "*.md", "*.toml", "requirements*.txt", "scripts/*.py", "scripts/*.sh",
    "src/lab1/*.py", "tests/*.py", "task*/data/**/README.md", "task*/data/**/.gitkeep",
    "task*/srinidhi/*.md", "task*/srinidhi/*.csv", "task*/srinidhi/config.json",
    "task*/srinidhi/*.py", "task*/srinidhi/src/*.py", "task*/srinidhi/src/*.ipynb",
    "task*/srinidhi/checkpoints/README.md", "task*/srinidhi/outputs/**/README.md",
    "task*/srinidhi/outputs/**/.gitkeep", "task*/srinidhi/data_processed/README.md",
    "task*/srinidhi/data_processed/rehearsal/**/*.json",
    "task*/srinidhi/data_processed/rehearsal/**/*.jsonl",
    "report/*.md", "report/*.csv", "reproducibility/README.md",
    "reproducibility/manifests/**/*.md", "reproducibility/manifests/**/*.json",
    "reproducibility/manifests/**/*.jsonl", "reproducibility/manifests/**/*.txt",
    "reproducibility/manifests/**/*.csv", "verification/*.json",
    "verification/*.xml", "verification/*.txt",
):
    paths.update(p for p in ROOT.glob(pattern) if p.is_file())
paths.add(ROOT / ".gitignore")
paths.add(ROOT / ".gitattributes")
# Preserve raw logs/metrics/plots; model weights are backed up separately.
# Real-data check weights remain local and are not final-model deliverables.
for run in ROOT.glob("reproducibility/raw_logs/srinidhi/*"):
    paths.update(p for p in run.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".txt", ".log", ".md", ".png"})
manifest = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)}
archive = DIST / "lab1-2342-prepared.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
    for path in sorted(paths):
        handle.write(path, "lab1-2342/" + str(path.relative_to(ROOT)))
    handle.writestr("lab1-2342/PACKAGE_MANIFEST.json", json.dumps({"purpose": "preparation, not final submission", "files": manifest}, indent=2) + "\n")
with zipfile.ZipFile(archive) as handle:
    if handle.testzip() is not None:
        raise RuntimeError("ZIP integrity check failed")
    for relative, info in manifest.items():
        content = handle.read("lab1-2342/" + relative)
        if hashlib.sha256(content).hexdigest() != info["sha256"]:
            raise RuntimeError(f"Archive hash mismatch: {relative}")
result = {"archive": str(archive.relative_to(ROOT)), "files": len(manifest)+1, "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "crc_and_content_hashes_verified": True, "contains_trained_final_weights": False}
(DIST / "package_verification.json").write_text(json.dumps(result, indent=2)+"\n")
(DIST / "SHA256SUMS.txt").write_text(result["sha256"] + "  " + archive.name + "\n")
print(json.dumps(result, indent=2))
if args.metric_weights:
    cache = ROOT / ".cache/metric_weights"
    weights = sorted(p for p in cache.rglob("*") if p.is_file())
    if not weights:
        raise FileNotFoundError("Run scripts/prefetch_metric_weights.py first")
    weight_archive = DIST / "lab1-evaluation-weights.zip"
    weight_manifest = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in weights}
    with zipfile.ZipFile(weight_archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for path in weights:
            handle.write(path, "lab1-2342/" + str(path.relative_to(ROOT)))
        handle.writestr("lab1-2342/EVALUATION_WEIGHTS_MANIFEST.json", json.dumps(weight_manifest, indent=2)+"\n")
    with zipfile.ZipFile(weight_archive) as handle:
        if handle.testzip() is not None:
            raise RuntimeError("Evaluation-weight ZIP integrity check failed")
        for relative, metadata in weight_manifest.items():
            if hashlib.sha256(handle.read("lab1-2342/" + relative)).hexdigest() != metadata["sha256"]:
                raise RuntimeError("Evaluation-weight archive checksum mismatch")
    weight_result = {"archive": str(weight_archive.relative_to(ROOT)), "bytes": weight_archive.stat().st_size, "sha256": hashlib.sha256(weight_archive.read_bytes()).hexdigest(), "purpose": "frozen pretrained evaluation networks only, not trained lab models", "crc_and_content_hashes_verified": True}
    (DIST / "evaluation_weights_verification.json").write_text(json.dumps(weight_result, indent=2)+"\n")
    with (DIST / "SHA256SUMS.txt").open("a") as stream:
        stream.write(weight_result["sha256"] + "  " + weight_archive.name + "\n")
    print(json.dumps(weight_result, indent=2))
