"""Transfer integrity checks without loading any checkpoint pickle."""
import importlib.util
import os
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("checkpoint_bundle", Path(__file__).resolve().parents[1] / "scripts/checkpoint_bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def run_folder(tmp_path):
    root = tmp_path / "paused"
    for name in ("mean", "cnn", "bigru"):
        folder = root / name / "checkpoints"
        folder.mkdir(parents=True)
        (folder / "last.pt").write_bytes((name * 12).encode())
    (root / "resolved_config.json").write_text('{"seed":2342}')
    (root / "vocabulary.json").write_text('["word"]')
    (root / "training_log.jsonl").write_text('{"step":1}\n')
    return root


def test_roundtrip_preserves_all_models_and_only_run_files(tmp_path):
    root = run_folder(tmp_path)
    (tmp_path / "private.env").write_text("not part of run")
    output = tmp_path / "portable.zip"
    bundle.package(root, output)
    assert bundle.verify(output) == {"verified": True, "files": 6, "run_name": "paused"}
    with zipfile.ZipFile(output) as archive:
        assert "private.env" not in archive.namelist()
        for path in root.rglob("*"):
            if path.is_file():
                assert archive.read("run/" + path.relative_to(root).as_posix()) == path.read_bytes()
    with pytest.raises(FileExistsError):
        bundle.package(root, output)
    with pytest.raises(ValueError, match="outside"):
        bundle.package(root, root / "self.zip")


def test_empty_and_symlink_runs_are_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="checkpoint"):
        bundle.package(empty, tmp_path / "empty.zip")
    root = run_folder(tmp_path)
    (root / "linked").symlink_to(empty, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        bundle.package(root, tmp_path / "linked.zip")
    assert not (tmp_path / "linked.zip").exists()


def test_modified_content_is_detected_even_when_size_and_mtime_preserved(tmp_path, monkeypatch):
    root = run_folder(tmp_path)
    changed = root / "training_log.jsonl"
    original_verify = bundle.verify

    def mutate_after_zip(archive):
        result = original_verify(archive)
        previous = changed.stat()
        changed.write_text('{"step":2}\n')
        os.utime(changed, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        return result

    monkeypatch.setattr(bundle, "verify", mutate_after_zip)
    with pytest.raises(RuntimeError, match="content changed"):
        bundle.package(root, tmp_path / "changed.zip")
    assert not (tmp_path / "changed.zip").exists()
    assert not list(tmp_path.glob(".checkpoint-bundle-*"))


@pytest.mark.parametrize("change", ["size", "mtime", "inventory"])
def test_changed_run_is_not_published(tmp_path, monkeypatch, change):
    root = run_folder(tmp_path)
    changed = root / "training_log.jsonl"
    original_verify = bundle.verify

    def mutate_after_zip(archive):
        result = original_verify(archive)
        if change == "size":
            changed.write_text("more training happened")
        elif change == "mtime":
            previous = changed.stat()
            os.utime(changed, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1000000))
        else:
            (root / "new.log").write_text("new output")
        return result

    monkeypatch.setattr(bundle, "verify", mutate_after_zip)
    with pytest.raises(RuntimeError, match="changed"):
        bundle.package(root, tmp_path / "changed.zip")
    assert not (tmp_path / "changed.zip").exists()


def test_tampered_zip_and_unsafe_path_are_rejected(tmp_path):
    root = run_folder(tmp_path)
    output = tmp_path / "original.zip"
    bundle.package(root, output)
    with zipfile.ZipFile(output) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries["run/mean/checkpoints/last.pt"] = b"changed"
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="hash/size mismatch"):
        bundle.verify(tampered)
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.pt", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        bundle.verify(unsafe)
