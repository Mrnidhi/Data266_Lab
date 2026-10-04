import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_srinidhi", Path(__file__).resolve().parents[1] / "scripts/prepare_srinidhi.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def package(tmp_path, monkeypatch):
    folder = tmp_path / prepare.PACKAGES
    folder.mkdir(parents=True)
    archive = folder / "part1.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Part 1/task1_llm/srinidhi/checkpoints/last.pt", b"selected checkpoint")
        handle.writestr("Part 1/task1_llm/srinidhi/data_processed/full/train.jsonl", b"frozen data")
        handle.writestr("Part 1/task1_llm/srinidhi/results.md", b"must not replace current writing")
    monkeypatch.setattr(prepare, "ARCHIVES", {1: (archive.name, prepare.sha256(archive))})
    return archive


def test_restore_preserves_stale_files_and_leaves_writing_alone(tmp_path, monkeypatch):
    package(tmp_path, monkeypatch)
    checkpoint = tmp_path / "task1_llm/srinidhi/checkpoints/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"previous checkpoint")
    result = checkpoint.parent.parent / "results.md"
    result.write_text("current writing")
    assert prepare.restore(tmp_path, [1]) == (2, 0)
    assert checkpoint.read_bytes() == b"selected checkpoint"
    backups = list(
        (tmp_path / "runs/pdf-repair-20261004").glob("*/task1_llm/srinidhi/checkpoints/last.pt")
    )
    assert len(backups) == 1 and backups[0].read_bytes() == b"previous checkpoint"
    assert result.read_text() == "current writing"
    assert prepare.restore(tmp_path, [1]) == (0, 2)


def test_wrong_archive_is_rejected_before_restoring(tmp_path, monkeypatch):
    archive = package(tmp_path, monkeypatch)
    with archive.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="git lfs pull"):
        prepare.restore(tmp_path, [1])
    assert not (tmp_path / "task1_llm").exists()


def test_external_destination_is_rejected(tmp_path, monkeypatch):
    package(tmp_path, monkeypatch)
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "task1_llm").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare.restore(tmp_path, [1])
    assert not list(external.iterdir())


def test_archive_path_cannot_escape_repo():
    with pytest.raises(ValueError, match="Unsafe archive path"):
        prepare.asset_path("Part 1/../outside.pt", 1)


def test_image_restore_skips_mac_metadata_and_preserves_originals(tmp_path, monkeypatch):
    folder = tmp_path / prepare.PACKAGES
    folder.mkdir(parents=True)
    archive = folder / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("dataset/monet_jpg/monet.jpg", b"source Monet")
        handle.writestr("dataset/photo_jpg/photo.jpg", b"source Photo")
        handle.writestr("__MACOSX/dataset/monet_jpg/._monet.jpg", b"metadata")
        handle.writestr("dataset/.DS_Store", b"metadata")
    monkeypatch.setattr(prepare, "ARCHIVES", {3: (archive.name, prepare.sha256(archive))})
    monkeypatch.setattr(prepare, "IMAGE_COUNTS", {"monet_jpg": 1, "photo_jpg": 1})
    image = tmp_path / "task3_gan/data/monet_jpg/monet.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"previous image")
    assert prepare.restore(tmp_path, [3]) == (2, 0)
    assert image.read_bytes() == b"source Monet"
    assert len(list((tmp_path / "task3_gan/data").rglob("*.jpg"))) == 2
    backups = list(
        (tmp_path / "runs/pdf-repair-20261004").glob("*/task3_gan/data/monet_jpg/monet.jpg")
    )
    assert len(backups) == 1 and backups[0].read_bytes() == b"previous image"
    assert not (tmp_path / "__MACOSX").exists()


def test_wrong_image_inventory_cannot_mix_datasets(tmp_path):
    entries = [zipfile.ZipInfo("dataset/monet_jpg/one.jpg")]
    with pytest.raises(ValueError, match="Expected 300"):
        prepare.check_image_inventory(tmp_path, entries)
    with pytest.raises(ValueError, match="Unsafe archive path"):
        prepare.asset_path("dataset/../outside.jpg", 3)


def test_extra_local_images_are_not_silently_used(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "IMAGE_COUNTS", {"monet_jpg": 1, "photo_jpg": 1})
    entries = [zipfile.ZipInfo(f"dataset/{domain}/one.jpg") for domain in prepare.IMAGE_COUNTS]
    unrelated = tmp_path / "task3_gan/data/monet_jpg/other.jpg"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"keep this image")
    with pytest.raises(ValueError, match="Unexpected images"):
        prepare.check_image_inventory(tmp_path, entries)
    assert unrelated.read_bytes() == b"keep this image"
