"""Bundle a paused training run for transfer, or verify its portable ZIP contents.

Pause training first. Change detection is a safeguard, not a training lock.
Datasets and source code outside the specified run directory are not included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

MANIFEST = "CHECKPOINT_BUNDLE_MANIFEST.json"
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def _digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def _stamp(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Only regular files are allowed; no symlinks: {path}")
    return value.st_size, value.st_mtime_ns, value.st_ino, value.st_dev


def _files(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {path}")
        if not path.is_dir():
            _stamp(path)
            result[path.relative_to(root).as_posix()] = path
    return result


def _unchanged(path, expected):
    if _stamp(path) != expected:
        raise RuntimeError(f"Run changed during packaging: {path}. Pause training and retry.")


def verify(archive):
    """Verify names, the complete file inventory, and streamed SHA256 hashes."""
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate paths in checkpoint bundle")
        for item in source.infolist():
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in item.filename
                    or str(path) != item.filename or item.is_dir()
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError(f"Unsafe bundle entry: {item.filename}")
        manifest = json.loads(source.read(MANIFEST))
        if manifest.get("format_version") != 1 or not isinstance(manifest.get("files"), dict):
            raise ValueError("Unsupported checkpoint bundle manifest")
        files = manifest["files"]
        if not files or set(names) != {MANIFEST, *(f"run/{name}" for name in files)}:
            raise ValueError("Bundle inventory differs from manifest")
        if not any(PurePosixPath(name).suffix in CHECKPOINT_SUFFIXES and info["bytes"] > 0
                   for name, info in files.items()):
            raise ValueError("Bundle contains no nonempty checkpoint")
        for relative, expected in files.items():
            info = source.getinfo(f"run/{relative}")
            with source.open(info) as handle:
                actual = _digest(handle)
            if info.file_size != expected["bytes"] or actual != expected["sha256"]:
                raise ValueError(f"Bundle hash/size mismatch: {relative}")
    return {"verified": True, "files": len(files), "run_name": manifest["run_name"]}


def package(run_dir, destination):
    root, destination = Path(run_dir).expanduser(), Path(destination).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Run directory must be an existing directory, not a symlink")
    root, destination = root.resolve(), destination.absolute()
    if destination.resolve().is_relative_to(root):
        raise ValueError("Write the bundle outside the run directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    paths = _files(root)
    stamps = {name: _stamp(path) for name, path in paths.items()}
    if not any(Path(name).suffix in CHECKPOINT_SUFFIXES and stamps[name][0] > 0 for name in paths):
        raise ValueError("Run contains no nonempty .pt, .pth, or .ckpt checkpoint")
    metadata = {}
    for name, path in paths.items():
        with path.open("rb") as handle:
            digest = _digest(handle)
        _unchanged(path, stamps[name])
        metadata[name] = {"bytes": stamps[name][0], "sha256": digest}
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint-bundle-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as target:
            for name, path in paths.items():
                _unchanged(path, stamps[name])
                target.write(path, f"run/{name}")
                _unchanged(path, stamps[name])
            target.writestr(MANIFEST, json.dumps({"format_version": 1, "run_name": root.name,
                "note": "Paused run snapshot; raw datasets outside this directory must be transferred separately.",
                "files": metadata}, indent=2) + "\n")
        verify(temporary)
        if set(_files(root)) != set(paths):
            raise RuntimeError("Run file inventory changed. Pause training and retry.")
        for name, path in paths.items():
            _unchanged(path, stamps[name])
            with path.open("rb") as handle:
                if _digest(handle) != metadata[name]["sha256"]:
                    raise RuntimeError(f"Run content changed: {path}. Pause training and retry.")
            _unchanged(path, stamps[name])
        # Same-directory hard link atomically publishes the complete file and refuses replacement.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"archive": str(destination), "files": len(paths), "bytes": destination.stat().st_size,
            "verified": True, "note": "Transfer datasets and the same repository revision separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Pause training, then bundle the entire run directory")
    create.add_argument("--run-dir", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify", help="Verify a received ZIP without extracting it")
    check.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = package(args.run_dir, args.output) if args.command == "create" else verify(args.archive)
    except (ValueError, OSError, RuntimeError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Checkpoint bundle failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
