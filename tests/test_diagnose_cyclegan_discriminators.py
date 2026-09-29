"""Tiny CPU fixtures verify diagnostics only; no class/GPU result is fabricated."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch


spec = importlib.util.spec_from_file_location("diagnose_cyclegan_discriminators",
    Path(__file__).resolve().parents[1] / "scripts/diagnose_cyclegan_discriminators.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.set_num_threads(1)
    data_paths = {"manifest_dir": str(tmp_path / "manifests")}
    Path(data_paths["manifest_dir"]).mkdir()
    for domain_offset, domain in enumerate(("photo", "monet")):
        folder = tmp_path / domain
        folder.mkdir()
        data_paths[f"{domain}_dir"] = str(folder)
        for split_offset, split in enumerate(("train", "val", "test")):
            names = []
            for index in range({"train": 3, "val": 2, "test": 1}[split]):
                name = f"{split}-{index}.png"
                rng = np.random.default_rng(domain_offset * 1000 + split_offset * 100 + index)
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / name)
                names.append(name)
            (Path(data_paths["manifest_dir"]) / f"{split}_{domain}.txt").write_text("\n".join(names) + "\n")
    config = {"mode": "full", "seed": 2342, "image_size": 32, "base_channels": 4,
              "residual_blocks": 1, "data": data_paths}
    _, manifest = diagnostics.cg.prepare_data(config)
    models = diagnostics.cg.build_models(config, "cpu")
    checkpoint = tmp_path / "fixture.pt"
    torch.save({"format_version": 1, "config": config,
                "state": {"global_step": 99, "manifest_fingerprint": manifest["manifest_fingerprint"]},
                "models": {name: model.state_dict() for name, model in models.items()}}, checkpoint)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"full": config}))
    return checkpoint, config_path, tmp_path / "diagnostics.json"


def test_patch_mse_is_computed_before_averaging_patches():
    result = diagnostics.patch_measurements(torch.tensor([[[[-1., 3.]]]]))
    assert result["patch_score_mean"] == 1.
    assert result["patch_mse_to_1"] == 4.
    assert result["patch_mse_to_0"] == 5.
    assert result["patch_count"] == 2


def test_fixed_subsets_are_reproducible_capped_and_include_entire_small_domain():
    assert diagnostics.subset_indices(30, 2342) == list(range(30))
    assert len(diagnostics.subset_indices(702, 2342)) == 128
    assert diagnostics.subset_indices(702, 2342) == diagnostics.subset_indices(702, 2342)
    assert diagnostics.subset_indices(702, 2342) != diagnostics.subset_indices(702, 2343)
    assert len(set(diagnostics.subset_indices(702, 2342))) == 128


def test_real_cpu_inference_preserves_checkpoint_and_skips_augmentation_and_test(tiny_checkpoint, monkeypatch):
    checkpoint, config, output = tiny_checkpoint
    before = diagnostics.digest(checkpoint)
    original_getitem = diagnostics.cg.ImageDomain.__getitem__
    original_build = diagnostics.cg.build_models
    seen, retained = [], {}
    def tracked_getitem(self, index):
        assert self.augment is False
        assert not self.records[index].name.startswith("test")
        seen.append(self.records[index].name)
        return original_getitem(self, index)
    def tracked_build(*args):
        result = original_build(*args)
        retained.update(result)
        return result
    monkeypatch.setattr(diagnostics.cg.ImageDomain, "__getitem__", tracked_getitem)
    monkeypatch.setattr(diagnostics.cg, "build_models", tracked_build)
    monkeypatch.setattr(diagnostics, "environment", lambda: {"fixture": True})
    report = diagnostics.diagnose(checkpoint, config, output, "cpu")
    assert seen and diagnostics.digest(checkpoint) == before
    assert report["source_checkpoint_unchanged"] and report["training_updates"] == 0
    assert report["augmentation"] is False and report["test_split_inference"] is False
    assert report["checkpoint_step"] == 99
    assert set(report["selection"]) == {"train_photo", "train_monet", "val_photo", "val_monet"}
    for name in ("D_photo", "D_monet"):
        groups = report["discriminators"][name]
        assert groups["real_train"]["count"] == 3
        assert groups["real_val"]["count"] == groups["fake_val"]["count"] == 2
        assert groups["real_val"]["target"] == 1 and groups["fake_val"]["target"] == 0
        assert set(groups["real_val"]["summary"]["patch_mse_to_1"]) == {"mean", "median", "q10", "q90"}
    assert all(parameter.grad is None and not parameter.requires_grad
               for model in retained.values() for parameter in model.parameters())
    assert json.loads(output.read_text())["checkpoint_sha256"] == before
    report_hash = diagnostics.digest(output)
    with pytest.raises(FileExistsError):
        diagnostics.diagnose(checkpoint, config, output, "cpu")
    assert diagnostics.digest(output) == report_hash


def test_wrong_manifest_blocks_diagnostics_before_inference(tiny_checkpoint):
    checkpoint, config, output = tiny_checkpoint
    saved = torch.load(checkpoint, weights_only=False)
    saved["state"]["manifest_fingerprint"] = "wrong"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="frozen class manifests"):
        diagnostics.diagnose(checkpoint, config, output, "cpu")
    assert not output.exists()


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], []])
def test_nonfinite_or_empty_values_are_rejected(values):
    with pytest.raises(ValueError):
        diagnostics.summary(values)
    with pytest.raises(FloatingPointError):
        diagnostics.patch_measurements(torch.tensor(values))
