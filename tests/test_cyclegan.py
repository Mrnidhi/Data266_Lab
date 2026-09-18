"""Small CPU checks for training correctness, provenance, and exact resume."""
import csv
import json
from pathlib import Path
import random

import numpy as np
import pytest
from lab1.common import task_config_path
import torch
from torch import nn
from PIL import Image

from lab1.cyclegan import (Generator, PatchDiscriminator, ReplayPool, build_models,
                          cycle_forward, generator_losses, prepare_data, run,
                          score_human_audit, export_checkpoint)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def smoke_config():
    location = task_config_path("cyclegan")
    config = json.loads(location.read_text())["smoke"]
    config["evaluate_after_run"] = False
    return config


def test_shapes_and_patch_receptive_field():
    model, disc = Generator(8, 1), PatchDiscriminator(8)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    y = model(x)
    assert y.shape == x.shape
    assert y.min() >= -1 and y.max() <= 1
    assert disc(y).shape == (2, 1, 2, 2)
    receptive, jump = 1, 1
    for layer in disc.layers:
        if isinstance(layer, nn.Conv2d):
            receptive += (layer.kernel_size[0] - 1) * jump
            jump *= layer.stride[0]
    assert receptive == 70


class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


def test_cycle_and_identity_direction_wiring():
    photo, monet = torch.zeros(1, 3, 4, 4), torch.ones(1, 3, 4, 4) * 10
    models = {"G_photo_to_monet": Add(2), "G_monet_to_photo": Add(3)}
    output = cycle_forward(models, photo, monet)
    assert torch.all(output["fake_monet"] == 2)
    assert torch.all(output["fake_photo"] == 13)
    assert torch.all(output["cycle_photo"] == 5)
    assert torch.all(output["cycle_monet"] == 15)
    assert torch.all(output["identity_photo"] == 3)
    assert torch.all(output["identity_monet"] == 12)


def test_identity_weight_is_absolute_five():
    photo = torch.zeros(1, 3, 4, 4)
    monet = torch.ones_like(photo)
    output = {"fake_monet": monet, "fake_photo": photo,
              "cycle_photo": photo, "cycle_monet": monet,
              "identity_photo": photo + 1, "identity_monet": monet + 1}
    losses = generator_losses({"D_photo": Add(1), "D_monet": Add(0)}, photo, monet, output,
                              cycle_weight=10, identity_weight=5)
    assert losses["generator_total"].item() == pytest.approx(10)


def test_replay_detaches_and_roundtrips():
    random.seed(2342)
    pool = ReplayPool(2)
    result = pool.query(torch.randn(3, 3, 4, 4, requires_grad=True))
    assert result.grad_fn is None and not result.requires_grad
    assert len(pool.images) == 2
    restored = ReplayPool(0)
    restored.load_state_dict(pool.state_dict())
    assert restored.capacity == 2
    assert all(torch.equal(a, b) for a, b in zip(pool.images, restored.images))


def test_full_cannot_use_synthetic_and_partial_paths_fail():
    config = smoke_config()
    config["mode"] = "full"
    with pytest.raises(ValueError, match="Actual class data required"):
        prepare_data(config)
    config["mode"] = "rehearsal"
    config["data"] = {"photo_dir": "missing"}
    with pytest.raises(ValueError, match="all three"):
        prepare_data(config)


def test_real_manifests_use_domain_relative_paths_and_reject_content_leakage(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for domain_number, domain in enumerate(("photo", "monet")):
        folder = tmp_path / domain
        folder.mkdir()
        for split_number, split in enumerate(("train", "val", "test")):
            filename = f"{split}.png"
            Image.new("RGB", (40, 40), color=(30 + 30 * domain_number, 20 * split_number, 60)).save(folder / filename)
            (manifests / f"{split}_{domain}.txt").write_text(filename + "\n")
    config = dict(smoke_config(), mode="full", allow_synthetic=False,
                  data={"photo_dir": str(tmp_path / "photo"), "monet_dir": str(tmp_path / "monet"), "manifest_dir": str(manifests)})
    data, provenance = prepare_data(config)
    assert provenance["class_results"] is True
    assert len(data) == 6
    assert data["test_photo"][0].shape == (3, 32, 32)
    (tmp_path / "photo" / "val.png").write_bytes((tmp_path / "photo" / "train.png").read_bytes())
    with pytest.raises(ValueError, match="Duplicate image content"):
        prepare_data(config)


def test_resume_matches_uninterrupted_all_networks_optimizers_and_pools(tmp_path):
    config = smoke_config()
    full = run(config, tmp_path / "continuous", "cpu")
    first = dict(config, max_steps=1)
    run(first, tmp_path / "resumed", "cpu")
    resumed = run(config, tmp_path / "resumed", "cpu", tmp_path / "resumed" / "last.pt")
    a = torch.load(tmp_path / "continuous" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    assert full["completed_updates"] == resumed["completed_updates"] == 2
    for name in a["models"]:
        assert all(torch.equal(a["models"][name][k], b["models"][name][k]) for k in a["models"][name])
    for name in a["optimizers"]:
        for index, state in a["optimizers"][name]["state"].items():
            for key, tensor in state.items():
                assert torch.equal(tensor, b["optimizers"][name]["state"][index][key])
    assert a["schedulers"] == b["schedulers"]
    for domain in ("photo", "monet"):
        assert all(torch.equal(x, y) for x, y in zip(a["replay_pools"][domain]["images"], b["replay_pools"][domain]["images"]))
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["rng"]["python"] == b["rng"]["python"]
    assert resumed["best_checkpoint"] is None
    with pytest.raises(ValueError, match="Synthetic rehearsal"):
        export_checkpoint(tmp_path / "resumed" / "last.pt", tmp_path, tmp_path / "absent.txt", tmp_path / "export", "cpu")


def test_smoke_evaluation_marks_optional_quality_metrics_unavailable(tmp_path):
    config = dict(smoke_config(), evaluate_after_run=True, max_steps=1)
    report = run(config, tmp_path, "cpu")
    assert report["class_results"] is False
    for direction in ("photo_to_monet", "monet_to_photo"):
        metrics = report["evaluation"]["directions"][direction]["metrics"]
        assert metrics["cycle_l1"]["status"] == "computed"
        for name in ("fid", "kid", "lpips_cycle", "content_cosine", "precision", "recall", "density", "coverage"):
            assert metrics[name]["status"] == "unavailable"
            assert metrics[name]["value"] is None


def test_human_agreement_requires_ratings_and_known_perfect_case(tmp_path):
    fields = ["sample_id", "direction", "style", "content", "artifacts"]
    paths = [tmp_path / "one.csv", tmp_path / "two.csv"]
    rows = [["s1", "photo_to_monet", "", "", ""], ["s2", "photo_to_monet", "", "", ""]]
    def write(values):
        for path in paths:
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(values)
    write(rows)
    with pytest.raises(ValueError, match="Actual human ratings"):
        score_human_audit(*paths)
    write([["s1", "photo_to_monet", 1, 2, 3], ["s2", "photo_to_monet", 5, 4, 2]])
    result = score_human_audit(*paths)
    assert result["groups"]["all"]["criteria"]["style"]["percent_agreement"] == 100
    assert result["groups"]["all"]["criteria"]["style"]["quadratic_weighted_kappa"] == 1
