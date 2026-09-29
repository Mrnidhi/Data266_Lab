"""Small CPU checks for training correctness, provenance, and exact resume."""
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import lab1.cyclegan as cyclegan
from lab1.common import task_config_path
import torch
from torch import nn
from PIL import Image

from lab1.cyclegan import (Generator, PatchDiscriminator, ReplayPool, build_models,
                          cycle_forward, generator_losses, prepare_data, run,
                          score_human_audit, export_checkpoint, restore_rng, discriminator_loss)


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


def test_bfloat16_losses_reduce_in_fp32_and_replay_serialization_is_independent():
    photo = torch.zeros(1, 3, 4, 4, dtype=torch.bfloat16)
    monet = torch.ones_like(photo)
    output = {"fake_monet": monet, "fake_photo": photo,
              "cycle_photo": photo, "cycle_monet": monet,
              "identity_photo": photo + 1, "identity_monet": monet + 1}
    losses = generator_losses({"D_photo": Add(1), "D_monet": Add(0)}, photo, monet, output)
    assert all(value.dtype == torch.float32 for value in losses.values())
    assert losses["generator_total"].item() == pytest.approx(10)
    assert discriminator_loss(Add(0), monet, photo).dtype == torch.float32
    pool = ReplayPool(2)
    pool.query(monet)
    state = pool.state_dict()
    assert state["images"][0].device.type == "cpu"
    assert state["images"][0].dtype == torch.bfloat16
    state["images"][0].zero_()
    assert pool.images[0].sum() > 0


@pytest.mark.parametrize("options,device,match", [
    ({"precision": "fp16"}, "cpu", "precision must"),
    ({"precision": "unknown"}, "cpu", "precision must"),
    ({"precision": "bf16"}, "cpu", "native BF16 support"),
    ({"replay_device": "unknown"}, "cpu", "replay_device must"),
])
def test_invalid_training_options_fail_before_creating_run(tmp_path, options, device, match):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=match):
        run(dict(smoke_config(), **options), output, device)
    assert not output.exists()


def test_bf16_requires_available_native_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="native BF16 support"):
        cyclegan._training_options({"precision": "bf16"}, "cuda")
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation: False)
    with pytest.raises(ValueError, match="native BF16 support"):
        cyclegan._training_options({"precision": "bf16"}, "cuda")


@pytest.mark.parametrize("source_count,destination_count", [(2, 1), (1, 2)])
def test_resume_restores_only_available_cuda_rng_devices(monkeypatch, capsys, source_count, destination_count):
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: destination_count)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: restored.append((device, state)))
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(), "cuda": [torch.tensor([i], dtype=torch.uint8) for i in range(source_count)]}
    restore_rng(state)
    assert [device for device, _ in restored] == list(range(min(source_count, destination_count)))
    assert all(torch.equal(value, state["cuda"][device]) for device, value in restored)
    assert "CUDA device count changed" in capsys.readouterr().out


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


def test_old_fp32_checkpoint_resumes_and_replay_placement_can_change(tmp_path):
    config = smoke_config()
    run(dict(config, max_steps=1), tmp_path / "old", "cpu")
    checkpoint = tmp_path / "old" / "last.pt"
    state = torch.load(checkpoint, weights_only=False)
    state["config"].pop("precision")
    state["config"].pop("replay_device")
    torch.save(state, checkpoint)
    report = run(dict(config, precision="fp32", replay_device="device"), tmp_path / "new", "cpu", checkpoint)
    assert report["completed_updates"] == 2
    assert report["precision"] == "fp32"
    assert report["replay_device"] == "device"
    state["config"]["precision"] = "bf16"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="preserve precision"):
        run(config, tmp_path / "changed", "cpu", checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA training check requires a GPU")
def test_cuda_bf16_training_replay_and_checkpoint_portability(tmp_path):
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("GPU has no native BF16 support")
    config = dict(smoke_config(), precision="bf16", replay_device="device")
    first = run(dict(config, max_steps=1), tmp_path / "gpu", "cuda")
    assert first["nan_events"] == 0
    checkpoint = torch.load(tmp_path / "gpu" / "last.pt", map_location="cpu", weights_only=False)
    assert all(value.device.type == "cpu" for pool in checkpoint["replay_pools"].values() for value in pool["images"])
    assert all(value.dtype == torch.float32 for model in checkpoint["models"].values() for value in model.values())
    resumed = run(dict(config, replay_device="cpu"), tmp_path / "resumed", "cuda", tmp_path / "gpu" / "last.pt")
    assert resumed["completed_updates"] == 2
    assert resumed["nan_events"] == 0


@pytest.fixture
def selected_best_run(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for domain_number, domain in enumerate(("photo", "monet")):
        folder = tmp_path / domain
        folder.mkdir()
        for split_number, split in enumerate(("train", "val", "test")):
            filename = f"{split}.png"
            Image.new("RGB", (40, 40), color=(30 + 30 * domain_number, 20 * split_number, 60)).save(folder / filename)
            (manifests / f"{split}_{domain}.txt").write_text(filename + "\n")
    config = dict(smoke_config(), mode="full", allow_synthetic=False, select_best=True, validation_every_epochs=1,
                  data={"photo_dir": str(tmp_path / "photo"), "monet_dir": str(tmp_path / "monet"), "manifest_dir": str(manifests)})
    def fake_evaluation(*args, **kwargs):
        return {"directions": {name: {"metrics": {"kid": {"status": "computed", "value": 0.2}}}
                               for name in cyclegan.DIRECTIONS}}
    monkeypatch.setattr(cyclegan, "evaluate_models", fake_evaluation)
    source = tmp_path / "source_run"
    run(dict(config, max_steps=1), source, "cpu")
    return config, source


def test_resume_into_fresh_directory_preserves_prior_best_without_improvement(tmp_path, selected_best_run):
    config, source = selected_best_run
    original = (source / "best.pt").read_bytes()
    destination = tmp_path / "new_platform"
    result = run(config, destination, "cpu", source / "last.pt")
    assert result["completed_updates"] == 2
    assert result["best_selection"]["step"] == 1
    assert result["best_checkpoint"] == str(destination / "best.pt")
    assert (destination / "best.pt").read_bytes() == original


def test_warm_start_resets_state_and_revalidates_baseline_without_modifying_source(tmp_path, selected_best_run, monkeypatch):
    config, source = selected_best_run
    source_path = source / "last.pt"
    saved = torch.load(source_path, map_location="cpu", weights_only=False)
    saved["state"].update(training_seconds=987., nan_events=99, peak_gpu_memory_bytes=12345)
    saved["rng"] = {"must_not_restore_source_rng": True}
    for optimizer in saved["optimizers"].values():
        optimizer["param_groups"][0]["lr"] = 0.
    torch.save(saved, source_path)
    original = source_path.read_bytes()
    evaluations = []
    def evaluate(*args, **kwargs):
        evaluations.append((args[5], args[6]))
        score = 0.1 if len(evaluations) == 1 else 0.3
        return {"directions": {name: {"metrics": {"kid": {"status": "computed", "value": score}}}
                               for name in cyclegan.DIRECTIONS}}
    monkeypatch.setattr(cyclegan, "evaluate_models", evaluate)
    destination = tmp_path / "fine_tune"
    recipe = dict(config, learning_rate=5e-5, identity_weight=2.5, cycle_weight=9.,
                  seed=2343, replay_size=3, epochs=4, constant_epochs=2, max_steps=1)
    result = run(recipe, destination, "cpu", warm_start=source_path)
    baseline = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    final = torch.load(destination / "last.pt", map_location="cpu", weights_only=False)
    assert baseline["state"]["global_step"] == 0
    assert baseline["state"]["training_seconds"] == baseline["state"]["nan_events"] == 0
    assert baseline["state"]["peak_gpu_memory_bytes"] == 0
    assert baseline["state"]["best_validation_score"] == pytest.approx(0.1)
    assert all(not optimizer["state"] for optimizer in baseline["optimizers"].values())
    assert all(optimizer["param_groups"][0]["lr"] == 5e-5 for optimizer in baseline["optimizers"].values())
    assert all(scheduler["last_epoch"] == 0 for scheduler in baseline["schedulers"].values())
    assert all(pool == {"capacity": 3, "images": []} for pool in baseline["replay_pools"].values())
    assert "torch" in baseline["rng"] and "must_not_restore_source_rng" not in baseline["rng"]
    for name, weights in saved["models"].items():
        assert all(torch.equal(value, baseline["models"][name][key]) for key, value in weights.items())
        assert any(not torch.equal(value, final["models"][name][key]) for key, value in weights.items())
    assert final["state"]["global_step"] == result["completed_updates"] == 1
    assert final["state"]["best_selection"]["step"] == result["best_selection"]["step"] == 0
    assert final["state"]["nan_events"] == 0
    assert [row["step"] for row in map(json.loads, (destination / "training_log.jsonl").read_text().splitlines())] == [1]
    initialization = result["initialization"]
    assert initialization == json.loads((destination / "initialization.json").read_text())
    assert initialization == final["state"]["initialization"]
    assert initialization["source_global_step"] == 1
    assert initialization["checkpoint_sha256"] == hashlib.sha256(original).hexdigest()
    assert initialization["source_config"] == saved["config"]
    assert initialization["source_best_validation_score"] == 0.2
    assert initialization["baseline_validation"]["value"] == 0.1
    assert evaluations == [("val", recipe["evaluation"])] * 2
    assert source_path.read_bytes() == original


def test_warm_start_requires_usable_validation_baseline_before_updates(tmp_path, selected_best_run, monkeypatch):
    config, source = selected_best_run
    def unavailable(*args, **kwargs):
        return {"directions": {name: {"metrics": {"kid": {"status": "unavailable", "value": None}}}
                               for name in cyclegan.DIRECTIONS}}
    monkeypatch.setattr(cyclegan, "evaluate_models", unavailable)
    destination = tmp_path / "missing_baseline"
    with pytest.raises(RuntimeError, match="finite validation KID"):
        run(config, destination, "cpu", warm_start=source / "last.pt")
    assert not (destination / "training_log.jsonl").exists()
    assert not (destination / "last.pt").exists()
    assert json.loads((destination / "initialization.json").read_text())["baseline_validation"]["status"] == "unavailable"


@pytest.mark.parametrize("changes,match", [
    ({"image_size": 36}, "architecture field image_size"),
    ({"base_channels": 16}, "architecture field base_channels"),
    ({"residual_blocks": 2}, "architecture field residual_blocks"),
    ({"synthetic_images": 5}, "data manifest mismatch"),
])
def test_warm_start_rejects_architecture_or_data_changes(tmp_path, changes, match):
    config = smoke_config()
    source = tmp_path / "source"
    run(dict(config, max_steps=1), source, "cpu")
    destination = tmp_path / "changed"
    with pytest.raises(ValueError, match=match):
        run(dict(config, **changes), destination, "cpu", warm_start=source / "last.pt")
    assert not (destination / "last.pt").exists()
    assert not (destination / "training_log.jsonl").exists()


def test_warm_start_and_resume_are_mutually_exclusive_and_source_directory_is_protected(tmp_path, monkeypatch):
    destination = tmp_path / "new"
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(smoke_config(), destination, "cpu", resume=tmp_path / "a.pt", warm_start=tmp_path / "b.pt")
    assert not destination.exists()
    with pytest.raises(ValueError, match="separate output directory"):
        run(smoke_config(), destination, "cpu", warm_start=destination / "last.pt")
    assert not destination.exists()
    monkeypatch.setattr("sys.argv", ["cyclegan", "train", "--output-dir", str(destination),
                                   "--resume", "old.pt", "--warm-start", "old.pt"])
    with pytest.raises(SystemExit) as error:
        cyclegan.main()
    assert error.value.code == 2


def test_warm_started_run_resume_matches_continuation_and_preserves_baseline(tmp_path, selected_best_run):
    config, source = selected_best_run
    recipe = dict(config, learning_rate=5e-5, identity_weight=2.5, epochs=4, constant_epochs=2, max_steps=2)
    uninterrupted = tmp_path / "uninterrupted_finetune"
    first = tmp_path / "first_finetune"
    continued = tmp_path / "continued_finetune"
    run(recipe, uninterrupted, "cpu", warm_start=source / "last.pt")
    run(dict(recipe, max_steps=1), first, "cpu", warm_start=source / "last.pt")
    result = run(recipe, continued, "cpu", resume=first / "last.pt")
    a = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(continued / "last.pt", map_location="cpu", weights_only=False)
    assert result["completed_updates"] == 2
    assert result["initialization"]["source_global_step"] == 1
    assert result["best_selection"]["step"] == 0
    assert (continued / "best.pt").read_bytes() == (first / "best.pt").read_bytes()
    for name in a["models"]:
        assert all(torch.equal(a["models"][name][key], b["models"][name][key]) for key in a["models"][name])
    for name in a["optimizers"]:
        for index, state in a["optimizers"][name]["state"].items():
            for key, tensor in state.items():
                assert torch.equal(tensor, b["optimizers"][name]["state"][index][key])
    assert a["schedulers"] == b["schedulers"]
    for domain in ("photo", "monet"):
        assert all(torch.equal(x, y) for x, y in zip(a["replay_pools"][domain]["images"], b["replay_pools"][domain]["images"]))
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["rng"]["python"] == b["rng"]["python"]
    with pytest.raises(ValueError, match="preserve identity_weight"):
        run(dict(recipe, identity_weight=1.), tmp_path / "changed_resume", "cpu", resume=first / "last.pt")


@pytest.mark.parametrize("problem", ["future", "recipe", "data", "initialization", "stale", "missing", "destination"])
def test_resume_rejects_inconsistent_best_checkpoint(tmp_path, selected_best_run, problem):
    config, source = selected_best_run
    path = source / "best.pt"
    candidate = torch.load(path, map_location="cpu", weights_only=False)
    destination = tmp_path / "new_platform"
    if problem == "missing":
        path.unlink()
    elif problem == "destination":
        destination.mkdir()
        (destination / "best.pt").write_bytes(b"different artifact")
    else:
        if problem == "future":
            candidate["state"]["global_step"] = 100
        elif problem == "recipe":
            candidate["config"]["learning_rate"] *= 2
        elif problem == "data":
            candidate["state"]["manifest_fingerprint"] = "different data"
        elif problem == "initialization":
            candidate["state"]["initialization"] = {"checkpoint_sha256": "different weights"}
        else:
            candidate["state"]["best_selection"]["value"] = 0.1
        torch.save(candidate, path)
    with pytest.raises((ValueError, FileExistsError), match="[Bb]est"):
        run(config, destination, "cpu", source / "last.pt")
    assert not (destination / "last.pt").exists()


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
