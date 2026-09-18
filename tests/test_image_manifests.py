import importlib.util
from pathlib import Path
import pytest
from PIL import Image
from lab1.cyclegan import prepare_data

spec = importlib.util.spec_from_file_location("prepare_image_manifests", Path(__file__).resolve().parents[1] / "scripts/prepare_image_manifests.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def test_manifest_helper_matches_training_loader_and_detects_duplicates(tmp_path):
    roots = {name: tmp_path / name for name in ("monet", "photo")}
    for domain_index, (domain, root) in enumerate(roots.items()):
        root.mkdir()
        for i in range(12):
            Image.new("RGB", (32, 32), (i * 13, domain_index * 101, 42)).save(root / f"{i}.png")
    helper.prepare(roots["monet"], roots["photo"], tmp_path / "manifests")
    config = {"mode": "full", "image_size": 32, "data": {"monet_dir": str(roots["monet"]), "photo_dir": str(roots["photo"]), "manifest_dir": str(tmp_path / "manifests")}}
    data, manifest = prepare_data(config)
    assert len(data) == 6
    assert sum(manifest["counts"].values()) == 24
    (roots["photo"] / "duplicate.png").write_bytes((roots["photo"] / "0.png").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        helper.prepare(roots["monet"], roots["photo"], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
