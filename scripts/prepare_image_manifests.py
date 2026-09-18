"""Create reproducible unpaired-image splits only when class-specific splits are absent."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
from PIL import Image


def prepare(monet_dir: Path, photo_dir: Path, output: Path, seed=2342, val_fraction=.1, test_fraction=.1):
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Manifest destination is not empty; preserve existing splits.")
    prepared = {}
    seen_hashes = set()
    for domain, source in [("monet", monet_dir), ("photo", photo_dir)]:
        if not source.is_dir():
            raise ValueError(f"Missing {domain} directory: {source}")
        files = sorted(p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        records = []
        for path in files:
            with Image.open(path) as image:
                image.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                raise ValueError(f"Duplicate image content detected: {path.name}; resolve before splitting.")
            seen_hashes.add(digest)
            records.append({"path": str(path.relative_to(source)), "sha256": digest})
        if len(records) < 10:
            raise ValueError(f"Need at least ten distinct {domain} images for this split helper.")
        rng = random.Random(seed + (0 if domain == "monet" else 1))
        rng.shuffle(records)
        val_n, test_n = max(1, int(len(records) * val_fraction)), max(1, int(len(records) * test_fraction))
        prepared[domain] = {"val": records[:val_n], "test": records[val_n:val_n+test_n],
                            "train": records[val_n+test_n:]}
    output.mkdir(parents=True)
    for domain, splits in prepared.items():
        for split, records in splits.items():
            (output / f"{split}_{domain}.txt").write_text("".join(r["path"] + "\n" for r in records))
    manifest = {"seed": seed, "method": "independent unpaired domain shuffle",
                "class_official_splits": False, "splits": prepared}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {domain: {split: len(rows) for split, rows in splits.items()} for domain, splits in prepared.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monet-dir", type=Path, required=True)
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2342)
    args = parser.parse_args()
    print(json.dumps(prepare(args.monet_dir, args.photo_dir, args.output, args.seed), indent=2))
