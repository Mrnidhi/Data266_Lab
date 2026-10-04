"""Restore verified data and checkpoints needed to reproduce the saved results."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

PACKAGES = Path("reproducibility/packages")
ARCHIVES = {
    1: (
        "parts1-2-20261002/Part1_Srinidhi_2342.zip",
        "879c648b101cd5ae38f5084ac0052c06832422d8e43ad146589b1596aba299bc",
    ),
    2: (
        "parts1-2-20261002/Part2_Srinidhi_2342.zip",
        "f9c11f4569efe6114ec8ed4551c3fdf27985dc7fa28ce65b500b49d58d973c89",
    ),
    3: (
        "part3-20261002/dataset.zip",
        "7529feac2b43278597ea795038f8e10aa673c0dc98d8de9f92071e236281e9cd",
    ),
}
MEMBERS = {1: "task1_llm/srinidhi", 2: "task2_sentiment/srinidhi"}
IMAGE_COUNTS = {"monet_jpg": 300, "photo_jpg": 7038}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def asset_path(name, part):
    if part == 3:
        image = PurePosixPath(name)
        if image.is_absolute() or ".." in image.parts:
            raise ValueError(f"Unsafe archive path: {name}")
        if (
            len(image.parts) == 3
            and image.parts[0] == "dataset"
            and image.parts[1] in IMAGE_COUNTS
            and image.suffix.lower() == ".jpg"
            and not image.name.startswith(".")
        ):
            return Path("task3_gan/data", image.parts[1], image.name)
        return None
    prefix = f"Part {part}/"
    if not name.startswith(prefix) or name.endswith("/"):
        return None
    relative = PurePosixPath(name[len(prefix) :])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    member = MEMBERS[part]
    path = relative.as_posix()
    data = path.startswith(f"{member}/data_processed/") and relative.suffix != ".md"
    member_checkpoint = path.startswith(f"{member}/checkpoints/")
    run_checkpoint = (
        path.startswith("reproducibility/raw_logs/srinidhi/") and "/checkpoints/" in path
    )
    checkpoint = relative.suffix == ".pt" and (member_checkpoint or run_checkpoint)
    return Path(relative) if data or checkpoint else None


def check_image_inventory(root, entries):
    paths = [asset_path(entry.filename, 3) for entry in entries]
    paths = [path for path in paths if path is not None]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image names in dataset archive.")
    for domain, count in IMAGE_COUNTS.items():
        expected = {path for path in paths if path.parent.name == domain}
        if len(expected) != count:
            raise ValueError(f"Expected {count} {domain} images in dataset archive.")
        directory = root / "task3_gan/data" / domain
        actual = {
            path.relative_to(root)
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        if actual - expected:
            raise ValueError(
                f"Unexpected images in {directory.relative_to(root)}; preserve them separately first."
            )


def restore(root, parts):
    archives = [(part, root / PACKAGES / ARCHIVES[part][0]) for part in parts]
    for part, archive in archives:
        if not archive.is_file() or sha256(archive) != ARCHIVES[part][1]:
            raise ValueError(f"Missing or incorrect {archive.name}; run git lfs pull first.")

    backup = (
        root / "runs/pdf-repair-20261004" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    (root / "runs").mkdir(exist_ok=True)
    restored = unchanged = 0
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=root / "runs") as temporary:
        temporary = Path(temporary)
        for part, archive in archives:
            with zipfile.ZipFile(archive) as package:
                if part == 3:
                    check_image_inventory(root, package.infolist())
                for entry in package.infolist():
                    relative = asset_path(entry.filename, part)
                    if relative is None:
                        continue
                    destination = root / relative
                    if any(path.is_symlink() for path in [destination, *destination.parents]):
                        raise ValueError(f"Refusing symlink destination: {relative}")
                    staged = temporary / "asset"
                    with package.open(entry) as source, staged.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    expected = sha256(staged)
                    if destination.is_file() and sha256(destination) == expected:
                        unchanged += 1
                        staged.unlink()
                        continue
                    if destination.exists():
                        preserved = backup / relative
                        preserved.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(destination, preserved)
                        if sha256(preserved) != sha256(destination):
                            raise ValueError(f"Backup verification failed: {relative}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    staged.replace(destination)
                    if sha256(destination) != expected:
                        raise ValueError(f"Restored file verification failed: {relative}")
                    restored += 1
                if part == 3:
                    for domain, count in IMAGE_COUNTS.items():
                        actual = len(list((root / "task3_gan/data" / domain).glob("*.jpg")))
                        if actual != count:
                            raise ValueError(
                                f"Expected {count} restored {domain} images, found {actual}."
                            )
                    print("Verified dataset domain counts, archive SHA-256 and every image hash.")
    print(f"Verified archives; restored {restored} files, {unchanged} already matched.")
    if backup.exists():
        print(f"Previous local files preserved in {backup.relative_to(root)}")
    return restored, unchanged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part", type=int, choices=(1, 2, 3), help="Default: prepare Parts 1 and 2."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    restore(root, [args.part] if args.part else [1, 2])


if __name__ == "__main__":
    main()
