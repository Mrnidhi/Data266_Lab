import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

from PIL import Image, PngImagePlugin
import pytest

spec = importlib.util.spec_from_file_location("prepare_part3_data", Path(__file__).resolve().parents[1] / "scripts/prepare_part3_data.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def image_bytes(domain, index, metadata=None):
    stream = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    if metadata:
        info.add_text("note", metadata)
    Image.new("RGB", (32, 32), (index * 13, domain * 101, 42)).save(stream, format="PNG", pnginfo=info)
    return stream.getvalue()


def archive_fixture(path, additions=()):
    with zipfile.ZipFile(path, "w") as archive:
        for domain_index, domain in enumerate(("monet", "photo")):
            for i in range(12):
                archive.writestr(f"dataset/dataset/{domain}_jpg/{i:02}.png", image_bytes(domain_index, i))
        archive.writestr("real_stats.npz", b"opaque class evaluator statistics preserved unchanged")
        for name, content in additions:
            archive.writestr(name, content)


def test_deduplication_split_provenance_and_portable_loader(tmp_path):
    archive = tmp_path / "class.zip"
    archive_fixture(archive, [
        ("dataset/dataset/photo_jpg/z-byte-duplicate.png", image_bytes(1, 0)),
        ("dataset/dataset/photo_jpg/z-pixel-duplicate.png", image_bytes(1, 1, "different metadata")),
    ])
    original = archive.read_bytes()
    root = tmp_path / "repo"
    receipt = helper.prepare(archive, root)
    assert archive.read_bytes() == original
    assert receipt["source_archive_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["raw_counts"] == {"monet": 12, "photo": 14}
    assert receipt["unique_counts"] == {"monet": 12, "photo": 12}
    assert receipt["counts"] == {f"{split}_{domain}": n for domain in ("monet", "photo") for split, n in (("train", 10), ("val", 1), ("test", 1))}
    raw = root / "task3_gan/data/photo_jpg"
    assert len(list(raw.glob("*.png"))) == 14
    snapshot = root / "reproducibility/manifests/srinidhi/part-c-source-snapshot"
    duplicates = json.loads((snapshot / "duplicates.json").read_text())["excluded_from_all_splits"]
    assert {row["reason"] for row in duplicates} == {"identical_bytes", "identical_decoded_rgb_pixels"}
    assert all(not Path(row["path"]).is_absolute() for row in json.loads((snapshot / "source_inventory.json").read_text())["images"])
    assert helper.prepare(archive, root) == receipt
    relocated = helper.prepare(archive, tmp_path / "another-platform")
    assert relocated["manifest_fingerprint"] == receipt["manifest_fingerprint"]
    assert relocated["manifest_sha256"] == receipt["manifest_sha256"]


def test_cross_domain_duplicate_is_rejected_before_publishing(tmp_path):
    archive = tmp_path / "cross-domain.zip"
    archive_fixture(archive, [("dataset/dataset/photo_jpg/z-cross-domain.png", image_bytes(0, 0, "new metadata"))])
    with pytest.raises(ValueError, match="Cross-domain duplicate"):
        helper.prepare(archive, tmp_path / "repo")
    assert not (tmp_path / "repo/task3_gan/data").exists()


def test_unsafe_zip_path_is_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    archive_fixture(archive, [("dataset/dataset/photo_jpg/../../outside.png", image_bytes(1, 0))])
    with pytest.raises(ValueError, match="Unsafe"):
        helper.prepare(archive, tmp_path / "repo")
    assert not (tmp_path / "repo/task3_gan/data").exists()


def test_existing_raw_file_is_never_overwritten(tmp_path):
    archive = tmp_path / "class.zip"
    archive_fixture(archive)
    root = tmp_path / "repo"
    destination = root / "task3_gan/data/photo_jpg/00.png"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing independent original")
    with pytest.raises(FileExistsError, match="Existing raw file differs"):
        helper.prepare(archive, root)
    assert destination.read_bytes() == b"existing independent original"
    assert not (root / "task3_gan/srinidhi/data_processed/part3_manifests").exists()
