"""Package integrity checks using small synthetic files."""

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_part3_push_package", ROOT / "scripts/build_part3_push_package.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    package_sources = source / "runner"
    package_sources.mkdir()
    for relative in (*builder.CODE, builder.EVALUATOR):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {relative}")
    for relative in builder.TOP_LEVEL:
        (package_sources / relative).write_text(f"fixture: {relative}")
    for name in ("init.pt", "arms.json", "dataset.zip", "inception.pth"):
        (source / name).write_bytes(name.encode())
    monkeypatch.setattr(builder, "ROOT", source)
    monkeypatch.setattr(builder, "PACKAGE_SOURCES", package_sources)
    monkeypatch.setattr(builder, "DATASET", source / "dataset.zip")
    monkeypatch.setattr(builder, "INCEPTION", source / "inception.pth")
    monkeypatch.setattr(
        builder, "EVALUATOR_SHA256", builder.sha256(source / builder.EVALUATOR)
    )
    return source


def test_built_archive_matches_the_manifest(tmp_path, sources):
    archive = builder.build(
        sources / "init.pt", sources / "arms.json", tmp_path / "out", "Part3_Test"
    )
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        manifest = json.loads(bundle.read("Part3_Test/inputs/MANIFEST.json"))
        for relative, expected in manifest["sha256"].items():
            assert (
                hashlib.sha256(bundle.read(f"Part3_Test/{relative}")).hexdigest()
                == expected
            )
        assert manifest["init_sha256"] == builder.sha256(sources / "init.pt")


@pytest.mark.parametrize("existing", ["Part3_Test", "Part3_Test.zip"])
def test_build_never_overwrites_existing_artifacts(tmp_path, sources, existing):
    out = tmp_path / "out"
    out.mkdir()
    original = out / existing
    original.write_text("original experiment")
    with pytest.raises(FileExistsError):
        builder.build(sources / "init.pt", sources / "arms.json", out, "Part3_Test")
    assert original.read_text() == "original experiment"


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../overwrite", "nested/name", "nested\\name"]
)
def test_package_name_cannot_escape_the_output_directory(tmp_path, name):
    with pytest.raises(ValueError, match="single directory"):
        builder.build(
            tmp_path / "init.pt", tmp_path / "arms.json", tmp_path / "out", name
        )


def test_changed_evaluator_is_rejected_before_creating_output(tmp_path, sources):
    (sources / builder.EVALUATOR).write_text("changed evaluator")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="notebook differs"):
        builder.build(sources / "init.pt", sources / "arms.json", out, "Part3_Test")
    assert not out.exists()
