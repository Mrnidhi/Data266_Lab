"""Preserve the class ZIP and freeze deduplicated CycleGAN image splits."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("image_split_helper", ROOT / "scripts/prepare_image_manifests.py")
split_helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_helper)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_preserving(source, destination):
    """An existing original or frozen manifest may be reused, never overwritten."""
    if destination.exists():
        if not destination.is_file() or sha256_file(source) != sha256_file(destination):
            raise FileExistsError(f"Existing file differs; preserve and review it: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def inspect_archive(archive, raw_stage):
    records, extras = [], []
    names = set()
    prefixes = {"monet": "dataset/dataset/monet_jpg/", "photo": "dataset/dataset/photo_jpg/"}
    with zipfile.ZipFile(archive) as zf:
        for member in sorted(zf.infolist(), key=lambda value: value.filename):
            name = member.filename
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or name in names:
                raise ValueError(f"Unsafe or repeated archive path: {name}")
            names.add(name)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive symlinks are not accepted: {name}")
            if member.is_dir():
                continue
            domain = next((key for key, prefix in prefixes.items() if name.startswith(prefix)), None)
            if domain is None and name != "real_stats.npz":
                raise ValueError(f"Unexpected archive entry; review before extraction: {name}")
            relative = name[len(prefixes[domain]):] if domain else name
            if not relative or (domain and Path(relative).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}):
                raise ValueError(f"Unsupported image entry: {name}")
            content = zf.read(member)
            record = {"archive_member": name, "path": relative, "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()}
            target = raw_stage / (f"{domain}_jpg" if domain else "") / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            if domain:
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
                with Image.open(io.BytesIO(content)) as image:
                    rgb = image.convert("RGB")
                    width, height = rgb.size
                    pixel_digest = hashlib.sha256(f"RGB:{width}:{height}:".encode() + rgb.tobytes()).hexdigest()
                    record.update(domain=domain, width=width, height=height, mode=image.mode,
                                  rgb_pixel_sha256=pixel_digest)
                records.append(record)
            else:
                extras.append(record)
    if not extras or not all(any(row["domain"] == domain for row in records) for domain in prefixes):
        raise ValueError("Class ZIP must contain Monet images, photo images, and real_stats.npz")
    return records, extras


def unique_records(records):
    """Keep the alphabetically first path for identical bytes or decoded RGB pixels."""
    hashes, pixels = {}, {}
    kept, duplicates = [], []
    for record in records:
        matching = hashes.get(record["sha256"]) or pixels.get(record["rgb_pixel_sha256"])
        if matching:
            if matching["domain"] != record["domain"]:
                raise ValueError(f"Cross-domain duplicate image: {matching['archive_member']} and {record['archive_member']}")
            duplicates.append({"domain": record["domain"], "excluded_path": record["path"],
                               "retained_path": matching["path"], "sha256": record["sha256"],
                               "retained_sha256": matching["sha256"],
                               "rgb_pixel_sha256": record["rgb_pixel_sha256"],
                               "reason": "identical_bytes" if record["sha256"] == matching["sha256"] else "identical_decoded_rgb_pixels"})
        else:
            kept.append(record)
        # Always point every byte representation back to the chosen canonical row.
        hashes[record["sha256"]] = matching or record
        pixels[record["rgb_pixel_sha256"]] = matching or record
    return kept, duplicates


def prepare(archive: Path, repo_root: Path = ROOT, seed: int = 2342):
    archive, repo_root = archive.resolve(), repo_root.resolve()
    raw_destination = repo_root / "task3_gan/data"
    manifest_destination = repo_root / "task3_gan/srinidhi/data_processed/part3_manifests"
    snapshot_destination = repo_root / "reproducibility/manifests/srinidhi/part-c-source-snapshot"
    archive_sha = sha256_file(archive)
    with tempfile.TemporaryDirectory(prefix="data266-part3-") as temporary:
        stage = Path(temporary)
        raw_stage = stage / "raw"
        records, extras = inspect_archive(archive, raw_stage)
        kept, duplicates = unique_records(records)
        for row in kept:
            source = raw_stage / f"{row['domain']}_jpg" / row["path"]
            target = stage / "unique" / row["domain"] / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        manifests = stage / "manifests"
        counts = split_helper.prepare(stage / "unique/monet", stage / "unique/photo", manifests, seed)
        snapshot = stage / "snapshot"
        shutil.copytree(manifests, snapshot)
        write_json(snapshot / "source_inventory.json", {"images": records, "other_files": extras})
        write_json(snapshot / "duplicates.json", {"canonical_selection": "alphabetically first archive path within domain",
                   "comparison": ["sha256 of original file bytes", "sha256 of RGB dimensions and decoded pixels"],
                   "cross_domain_duplicates": "rejected", "excluded_from_all_splits": duplicates})

        # Check every existing destination before publishing any new raw file.
        for source in raw_stage.rglob("*"):
            destination = raw_destination / source.relative_to(raw_stage)
            if source.is_file() and destination.exists() and sha256_file(source) != sha256_file(destination):
                raise FileExistsError(f"Existing raw file differs; preserve and review it: {destination}")
        for source in manifests.iterdir():
            destination = manifest_destination / source.name
            if destination.exists() and sha256_file(source) != sha256_file(destination):
                raise FileExistsError(f"Existing manifest differs; preserve and review it: {destination}")
        for source in raw_stage.rglob("*"):
            if source.is_file():
                copy_preserving(source, raw_destination / source.relative_to(raw_stage))
        for source in manifests.iterdir():
            copy_preserving(source, manifest_destination / source.name)

        sys.path.insert(0, str(ROOT / "src"))
        from lab1.cyclegan import prepare_data
        loader_config = {"mode": "full", "image_size": 256, "data": {
            "monet_dir": str(raw_destination / "monet_jpg"), "photo_dir": str(raw_destination / "photo_jpg"),
            "manifest_dir": str(manifest_destination)}}
        datasets, loader_manifest = prepare_data(loader_config)
        expected_counts = {f"{split}_{domain}": count for domain, splits in counts.items() for split, count in splits.items()}
        if loader_manifest["counts"] != expected_counts:
            raise AssertionError("Training loader counts differ from frozen split counts")
        # Decode through the actual loader, including its augmentation/normalization.
        for key, dataset in datasets.items():
            item = dataset[0]
            if tuple(item.shape) != (3, 256, 256) or not (-1 <= float(item.min()) <= float(item.max()) <= 1):
                raise AssertionError(f"Invalid model input for {key}")
        receipt = {"source_archive_filename": archive.name, "source_archive_bytes": archive.stat().st_size,
                   "source_archive_sha256": archive_sha, "source_archive_modified": False,
                   "class_official_splits": False, "seed": seed,
                   "split_policy": "After exact deduplication, independently shuffle Monet with seed and photos with seed+1. Validation and test each receive floor(10%); training receives the remainder.",
                   "raw_counts": dict(Counter(row["domain"] for row in records)),
                   "unique_counts": dict(Counter(row["domain"] for row in kept)),
                   "excluded_duplicate_counts": dict(Counter(row["domain"] for row in duplicates)),
                   "all_raw_images_preserved": True, "counts": expected_counts,
                   "manifest_fingerprint": loader_manifest["manifest_fingerprint"],
                   "manifest_sha256": {path.name: sha256_file(path) for path in sorted(manifests.iterdir())},
                   "data_paths": {"monet_dir": str((raw_destination / "monet_jpg").relative_to(repo_root)),
                                  "photo_dir": str((raw_destination / "photo_jpg").relative_to(repo_root)),
                                  "manifest_dir": str(manifest_destination.relative_to(repo_root))},
                   "validation": {"actual_training_loader": "passed", "six_domain_split_tensors": "passed", "cross_domain_duplicates": 0},
                   "real_stats_usage": "Preserved unchanged; class scoring protocol must establish its appropriate use. Held-out evaluation uses the frozen validation/test split."}
        if sha256_file(archive) != archive_sha:
            raise AssertionError("Source ZIP changed during preparation")
        write_json(snapshot / "preparation_receipt.json", receipt)
        for source in snapshot.iterdir():
            copy_preserving(source, snapshot_destination / source.name)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=2342)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.repo_root, args.seed), indent=2))
