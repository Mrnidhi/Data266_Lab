"""CPU-only checks for the optional Monet discriminator translation experiment."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lab1 import cyclegan as cg
from lab1.common import task_config_path


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def config(**changes):
    recipe = json.loads(task_config_path("cyclegan").read_text())["smoke"]
    recipe.update(evaluate_after_run=False, synthetic_images=2, replay_size=1,
                  epochs=2, constant_epochs=1, max_steps=4, monet_translation_ratio=.0625)
    recipe.update(changes)
    return recipe


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("ratio", [0, 0.0])
def test_disabled_translation_is_identity_and_consumes_no_rng(ratio):
    image = torch.ones(2, 3, 8, 8, requires_grad=True)
    before = cg.rng_state()
    assert cg.translation_augment(image, ratio) is image
    assert_nested_equal(before, cg.rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_translation_uses_zero_border_and_preserves_gradients(monkeypatch, dtype):
    # At 4x4 with ratio .25 the permitted shifts are -1, 0, and +1.
    shifts = iter([1, -1])

    def chosen_offset(low, high, size, device):
        assert (low, high, size, device) == (-1, 2, (1, 1, 1), "cpu")
        return torch.full(size, next(shifts), dtype=torch.long)

    monkeypatch.setattr(torch, "randint", chosen_offset)
    image = torch.arange(1, 17, dtype=dtype).reshape(1, 1, 4, 4).requires_grad_()
    actual = cg.translation_augment(image, .25)
    expected = torch.tensor([[0, 5, 6, 7], [0, 9, 10, 11],
                             [0, 13, 14, 15], [0, 0, 0, 0]], dtype=dtype).reshape_as(image)
    assert actual.shape == image.shape and actual.device == image.device and actual.dtype == dtype
    assert torch.equal(actual, expected)
    actual.float().sum().backward()
    expected_gradient = torch.zeros_like(image)
    expected_gradient[:, :, 1:, :3] = 1
    assert torch.equal(image.grad, expected_gradient)


def test_translation_cpu_rng_is_reproducible_without_changing_data_rng():
    image = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).reshape(2, 3, 32, 32)
    before = cg.rng_state()
    first = cg.translation_augment(image, .0625)
    after = cg.rng_state()
    assert before["python"] == after["python"]
    assert_nested_equal(before["numpy"], after["numpy"])
    assert not torch.equal(before["torch"], after["torch"])
    cg.restore_rng(before)
    assert torch.equal(first, cg.translation_augment(image, .0625))
    assert_nested_equal(after, cg.rng_state())


class RecordingDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.25))
        self.inputs = []

    def forward(self, image):
        self.inputs.append(image.detach().clone())
        return self.scale * image


def test_only_monet_generator_adversarial_input_is_augmented(monkeypatch):
    photo, monet = torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8)
    fake_monet = torch.full_like(monet, 2, requires_grad=True)
    output = {"fake_monet": fake_monet, "fake_photo": photo + 3,
              "cycle_photo": photo + 1, "cycle_monet": monet + 1,
              "identity_photo": photo + 2, "identity_monet": monet + 2}
    before = {key: value.detach().clone() for key, value in output.items()}
    models = {"D_photo": RecordingDiscriminator(), "D_monet": RecordingDiscriminator()}
    baseline = cg.generator_losses(models, photo, monet, output)
    monkeypatch.setattr(cg, "translation_augment", lambda image, ratio=0.: image + 1 if ratio else image)
    augmented = cg.generator_losses(models, photo, monet, output, monet_translation_ratio=.0625)
    assert torch.equal(models["D_monet"].inputs[-1], before["fake_monet"] + 1)
    assert torch.equal(models["D_photo"].inputs[-1], before["fake_photo"])
    assert not torch.equal(baseline["gan_photo_to_monet"], augmented["gan_photo_to_monet"])
    for name in ("gan_monet_to_photo", "cycle_photo_l1", "cycle_monet_l1",
                 "identity_photo_l1", "identity_monet_l1"):
        assert torch.equal(baseline[name], augmented[name])
    for key, value in output.items():
        assert torch.equal(value, before[key])
    augmented["gan_photo_to_monet"].backward()
    assert fake_monet.grad is not None and fake_monet.grad.abs().sum() > 0


def test_discriminator_augments_real_and_fake_but_detaches_generator(monkeypatch):
    seen = []

    def augmentation(image, ratio=0.):
        seen.append((image.requires_grad, ratio))
        return image + 1 if ratio else image

    monkeypatch.setattr(cg, "translation_augment", augmentation)
    real = torch.ones(1, 3, 8, 8)
    fake = torch.full_like(real, 2, requires_grad=True)
    discriminator = RecordingDiscriminator()
    loss = cg.discriminator_loss(discriminator, real, fake, translation_ratio=.0625)
    loss.backward()
    assert seen == [(False, .0625), (False, .0625)]
    assert torch.equal(discriminator.inputs[0], real + 1)
    assert torch.equal(discriminator.inputs[1], fake.detach() + 1)
    assert fake.grad is None and discriminator.scale.grad is not None


def test_training_routes_augmentation_to_monet_and_keeps_raw_replay(tmp_path, monkeypatch):
    originals = {name: getattr(cg, name) for name in ("build_models", "cycle_forward", "generator_losses", "discriminator_loss")}
    query = cg.ReplayPool.query
    generated, replay_inputs, routed = {}, [], []

    def build_models(*args, **kwargs):
        models = originals["build_models"](*args, **kwargs)
        for name, model in models.items():
            model.test_name = name
        return models

    def forward(*args, **kwargs):
        output = originals["cycle_forward"](*args, **kwargs)
        generated.update({name: output[f"fake_{name}"].detach().clone() for name in ("photo", "monet")})
        return output

    def generator_losses(models, photo, monet, output, cycle_weight=10., identity_weight=5., monet_translation_ratio=0.):
        routed.append(("generator", monet_translation_ratio))
        return originals["generator_losses"](models, photo, monet, output, cycle_weight, identity_weight, monet_translation_ratio)

    def discriminator_loss(discriminator, real, fake, translation_ratio=0.):
        routed.append((discriminator.test_name, translation_ratio))
        return originals["discriminator_loss"](discriminator, real, fake, translation_ratio)

    def pool_query(pool, batch):
        replay_inputs.append(batch.detach().clone())
        return query(pool, batch)

    monkeypatch.setattr(cg, "build_models", build_models)
    monkeypatch.setattr(cg, "cycle_forward", forward)
    monkeypatch.setattr(cg, "generator_losses", generator_losses)
    monkeypatch.setattr(cg, "discriminator_loss", discriminator_loss)
    monkeypatch.setattr(cg.ReplayPool, "query", pool_query)
    cg.run(config(max_steps=1), tmp_path / "routing", "cpu")
    assert routed == [("generator", .0625), ("D_photo", 0.), ("D_monet", .0625)]
    assert len(replay_inputs) == 2
    assert torch.equal(replay_inputs[0], generated["photo"])
    assert torch.equal(replay_inputs[1], generated["monet"])
    saved = torch.load(tmp_path / "routing" / "last.pt", map_location="cpu", weights_only=False)
    for name in ("photo", "monet"):
        assert torch.equal(saved["replay_pools"][name]["images"][0], generated[name])


def test_augmented_resume_matches_continuous_training_and_rng(tmp_path):
    recipe = config()
    cg.run(recipe, tmp_path / "continuous", "cpu")
    cg.run(dict(recipe, max_steps=2), tmp_path / "resumed", "cpu")
    cg.run(recipe, tmp_path / "resumed", "cpu", resume=tmp_path / "resumed" / "last.pt")
    continuous = torch.load(tmp_path / "continuous" / "last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(tmp_path / "resumed" / "last.pt", map_location="cpu", weights_only=False)
    for key in ("models", "optimizers", "schedulers", "replay_pools", "rng", "config"):
        assert_nested_equal(continuous[key], resumed[key])
    assert continuous["state"]["global_step"] == resumed["state"]["global_step"] == 4
    assert resumed["config"]["monet_translation_ratio"] == .0625


def test_resume_rejects_changed_augmentation_and_accepts_legacy_disabled(tmp_path):
    source = tmp_path / "source"
    cg.run(config(max_steps=1), source, "cpu")
    with pytest.raises(ValueError, match="preserve monet_translation_ratio"):
        cg.run(config(monet_translation_ratio=0.), tmp_path / "changed", "cpu", resume=source / "last.pt")
    disabled = tmp_path / "disabled"
    cg.run(config(max_steps=1, monet_translation_ratio=0.), disabled, "cpu")
    saved = torch.load(disabled / "last.pt", map_location="cpu", weights_only=False)
    saved["config"].pop("monet_translation_ratio")
    torch.save(saved, disabled / "last.pt")
    result = cg.run(config(max_steps=2, monet_translation_ratio=0.), tmp_path / "legacy", "cpu", resume=disabled / "last.pt")
    assert result["completed_updates"] == 2 and result["monet_translation_ratio"] == 0.


@pytest.mark.parametrize("ratio", [-.1, 1.1, float("nan"), float("inf"), True, "0.0625"])
def test_invalid_ratio_rejected_before_run_creation(tmp_path, ratio):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="monet_translation_ratio"):
        cg.run(config(monet_translation_ratio=ratio), output, "cpu")
    assert not output.exists()


def test_metric_images_are_enumerated_once_without_case_variant_globs(tmp_path, monkeypatch):
    (tmp_path / "second.PNG").write_bytes(b"fixture")
    (tmp_path / "first.png").write_bytes(b"fixture")
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "subdirectory.png").mkdir()
    calls = []

    def features(files, **kwargs):
        calls.append((files, kwargs))
        return np.ones((len(files), 3))

    def forbidden_folder_glob(*args, **kwargs):
        pytest.fail("Case-variant folder globbing can double Windows feature rows")

    monkeypatch.setattr(Path, "glob", forbidden_folder_glob)
    fid = SimpleNamespace(get_files_features=features, get_folder_features=forbidden_folder_glob)
    actual = cg._metric_image_features(fid, tmp_path, 2, num_workers=0, mode="clean")
    assert actual.shape == (2, 3)
    assert calls == [([str((tmp_path / name).resolve()) for name in ("first.png", "second.PNG")],
                      {"num_workers": 0, "mode": "clean"})]
    with pytest.raises(ValueError, match="Expected 3 metric images"):
        cg._metric_image_features(fid, tmp_path, 3)
    assert len(calls) == 1


@pytest.mark.parametrize("features,match", [
    (np.ones((4, 3)), "Expected 2 feature rows"),
    (np.ones(2), "Expected 2 feature rows"),
    (np.empty((2, 0)), "Expected 2 feature rows"),
    (np.full((2, 3), np.nan), "Non-finite metric features"),
])
def test_metric_feature_counts_and_finiteness_are_required(tmp_path, features, match):
    for name in ("a.png", "b.PNG"):
        (tmp_path / name).write_bytes(b"fixture")
    fid = SimpleNamespace(get_files_features=lambda *args, **kwargs: features)
    with pytest.raises(ValueError, match=match):
        cg._metric_image_features(fid, tmp_path, 2)
