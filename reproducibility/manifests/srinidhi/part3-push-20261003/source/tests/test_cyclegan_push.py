"""CPU-only checks for the warm-start leaderboard fine-tuning trainer."""
from dataclasses import replace
import json
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from PIL import Image

from lab1 import cyclegan as cg
from lab1 import cyclegan_push as cp

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "reproducibility/packages/part3-20261002/Part3_Evaluation_Script.ipynb"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def dataset(tmp_path, monet=4, photo=6, size=40):
    rng = np.random.default_rng(0)
    for domain, count in (("monet", monet), ("photo", photo)):
        folder = tmp_path / "data" / f"{domain}_jpg"
        folder.mkdir(parents=True)
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(folder / f"{index:04x}.jpg")
        (folder / "README.md").write_text("not an image")
    return tmp_path / "data"


def init_checkpoint(tmp_path):
    config = {"image_size": 64, "base_channels": 2, "residual_blocks": 1}
    torch.manual_seed(5)
    models = cg.build_models(config, "cpu")
    path = tmp_path / "init.pt"
    torch.save({"format_version": 1, "config": config, "state": {"global_step": 7},
                "models": {k: v.state_dict() for k, v in models.items()}}, path)
    return path


def recipe(**changes):
    base = cp.PushConfig(steps=6, eval_every=3, checkpoint_every=2, log_every=1, extra_scales=1,
                         d_warmup_steps=2, aug_monet="color,translation,cutout", aug_photo="translation",
                         flip_equivariance=0.5, cycle_start=10.0, cycle_end=4.0, identity_start=5.0,
                         identity_end=0.0, weight_ramp_steps=4, replay_size=2, learning_rate=2e-4)
    return replace(base, **changes)


def fake_scorer(generators):
    # Deterministic, consumes no RNG; lower is "better" for larger mean weights.
    value = float(sum(next(g.parameters()).detach().double().mean() for g in generators.values()))
    return {"fid_a2b": 90 - value, "fid_b2a": 95 - value, "mifid_a2b": .4, "mifid_b2a": .4,
            "fid": 92.5 - value, "mifid": .4, "composite": (92.9 - value) / 2}


def test_empty_policy_is_identity_without_rng_and_unknown_policy_fails():
    image = torch.randn(1, 3, 8, 8)
    before = torch.get_rng_state()
    assert cp.diff_augment(image, "") is image
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match="Unknown DiffAugment"):
        cp.diff_augment(image, "color,rotate")
    with pytest.raises(ValueError, match="Unknown DiffAugment"):
        cp.PushConfig(aug_monet="blur")


@pytest.mark.parametrize("policy", ["color", "translation", "cutout", "color,translation,cutout"])
def test_diffaugment_keeps_shape_and_passes_gradients(policy):
    image = torch.rand(2, 3, 16, 16, requires_grad=True)
    augmented = cp.diff_augment(image * 2 - 1, policy)
    assert augmented.shape == image.shape
    augmented.sum().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


def test_cutout_zeroes_a_square_and_translation_pads_with_zero(monkeypatch):
    image = torch.ones(1, 1, 8, 8)
    out = cp._cutout(image, ratio=0.5)
    assert int((out == 0).sum()) <= 16 and int((out == 0).sum()) >= 4
    shifted = cp._translation(image, ratio=0.25)
    assert shifted.shape == image.shape and set(shifted.unique().tolist()) <= {0.0, 1.0}


def test_multiscale_scale_zero_is_the_warm_started_patchgan():
    torch.manual_seed(0)
    base = cg.PatchDiscriminator(4)
    reference = cg.PatchDiscriminator(4)
    reference.load_state_dict(base.state_dict())
    model = cp.MultiScaleDiscriminator(base, 2, 4)
    x = torch.randn(1, 3, 128, 128)
    outputs = model(x)
    assert len(outputs) == 3
    assert torch.equal(outputs[0], reference(x))
    assert outputs[1].shape[-1] < outputs[0].shape[-1] and outputs[2].shape[-1] < outputs[1].shape[-1]


def test_schedules_hit_their_endpoints():
    assert cp.ramp(10, 4, 0, 4) == 10 and cp.ramp(10, 4, 2, 4) == 7 and cp.ramp(10, 4, 4, 4) == 4
    assert cp.ramp(10, 4, 0, 0) == 4
    assert cp.lr_factor(0, 10, .5) == 1 and cp.lr_factor(4, 10, .5) == 1
    assert cp.lr_factor(5, 10, .5) == 1 and cp.lr_factor(9, 10, .5) == pytest.approx(.2)
    assert cp.lr_factor(10, 10, .5) == 0


def test_config_rejects_unsafe_values():
    for bad in ({"cycle_end": 0.0}, {"precision": "fp16"}, {"data": "test"}, {"steps": 0},
                {"extra_scales": 4}, {"jpeg_subsampling": 3}, {"learning_rate": float("nan")}):
        with pytest.raises(ValueError):
            cp.PushConfig(**bad)


def test_fid_arithmetic_matches_the_supplied_notebook():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {}
    exec("import numpy as np\nimport scipy.linalg\n" + "".join(notebook["cells"][6]["source"]), namespace)
    rng = np.random.default_rng(1)
    real, generated = rng.normal(size=(40, 12)), rng.normal(.3, 1.2, size=(40, 12))
    fid, mifid = cp.fid_mifid(real, generated)
    expected = namespace["frechet_distance"](real.mean(0), np.cov(real, rowvar=False),
                                             generated.mean(0), np.cov(generated, rowvar=False))
    assert fid == expected
    from scipy.spatial.distance import cosine
    assert mifid == float(np.mean([cosine(real[i], generated[i]) for i in range(40)]))


def test_image_listing_matches_evaluator_semantics(tmp_path):
    data = dataset(tmp_path, monet=3, photo=3)
    (data / "monet_jpg" / "zz.PNG").write_bytes((data / "monet_jpg" / "0000.jpg").read_bytes())
    names = [p.name for p in cp.image_paths(data / "monet_jpg")]
    assert names == sorted(names) and "README.md" not in names and len(names) == 4


def test_training_resume_matches_continuous_and_keeps_best(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    push = recipe()
    full = cp.run(push, start, tmp_path / "full", "cpu", data, tmp_path, scorer=fake_scorer)
    assert full["status"] == "completed" and full["completed_updates"] == 6
    partial = cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, stop_after=3)
    assert partial["status"] == "partial" and partial["completed_updates"] == 3
    resumed = cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, resume=True)
    assert resumed["status"] == "completed"
    one = torch.load(tmp_path / "full/last.pt", weights_only=False)
    two = torch.load(tmp_path / "split/last.pt", weights_only=False)
    for key in ("models", "optimizers", "replay_pools", "ema", "rng"):
        equal(one[key], two[key])
    equal(one["state"]["history"], two["state"]["history"])
    assert [h["step"] for h in one["state"]["history"]] == [0, 3, 6]
    best = torch.load(tmp_path / "full/best.pt", weights_only=False)
    composites = [h["composite"] for h in one["state"]["history"]]
    assert best["class_scores"]["composite"] == min(composites)
    assert best["inference_only"] and set(best["models"]) == set(cp.GENERATORS + cp.DISCRIMINATORS)
    assert any(k.startswith("scales.1.") for k in best["models"]["D_monet"])


def test_warm_start_is_exact_and_discriminator_scale_zero_is_preserved(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    cp.run(recipe(steps=1, d_warmup_steps=0, eval_every=1), start, tmp_path / "run", "cpu", data, tmp_path,
           scorer=fake_scorer, stop_after=0)
    source = torch.load(start, weights_only=False)["models"]
    best = torch.load(tmp_path / "run/best.pt", weights_only=False)
    for name in cp.GENERATORS:
        equal(best["models"][name], source[name])
    for key, value in source["D_monet"].items():
        assert torch.equal(best["models"]["D_monet"]["scales.0." + key], value)


def test_refuses_existing_output_and_changed_recipe_on_resume(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    push = recipe(steps=2, eval_every=2)
    cp.run(push, start, tmp_path / "run", "cpu", data, tmp_path, scorer=fake_scorer, stop_after=1)
    with pytest.raises(FileExistsError):
        cp.run(push, start, tmp_path / "run", "cpu", data, tmp_path, scorer=fake_scorer)
    with pytest.raises(ValueError, match="identical push configuration"):
        cp.run(replace(push, learning_rate=1e-4), start, tmp_path / "run", "cpu", data, tmp_path,
               scorer=fake_scorer, resume=True)


def test_train_split_uses_only_manifest_training_images(tmp_path):
    data = dataset(tmp_path)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "train_photo.txt").write_text("0000.jpg\n0001.jpg\n")
    (manifests / "train_monet.txt").write_text("0002.jpg\n")
    paths = cp.training_paths("train_split", data, manifests)
    assert [p.name for p in paths["photo"]] == ["0000.jpg", "0001.jpg"]
    assert [p.name for p in paths["monet"]] == ["0002.jpg"]
    assert len(cp.training_paths("all", data, manifests)["photo"]) == 6


def test_raw_and_ema_are_both_scored_and_best_saves_the_winning_variant(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    push = recipe(steps=4, eval_every=2, eval_variants="ema,raw", ema_beta=0.9)

    def raw_wins(generators):
        # Raw weights move faster than the EMA copy, so their mean differs; score by distance from init.
        value = float(sum(next(g.parameters()).detach().double().abs().mean() for g in generators.values()))
        return {"fid_a2b": 90 - value, "fid_b2a": 95 - value, "mifid_a2b": .4, "mifid_b2a": .4,
                "fid": 92.5 - value, "mifid": .4, "composite": (92.9 - value) / 2}

    summary = cp.run(push, start, tmp_path / "run", "cpu", data, tmp_path, scorer=raw_wins)
    last = torch.load(tmp_path / "run/last.pt", weights_only=False)
    history = last["state"]["history"]
    assert [(h["step"], h["variant"]) for h in history] == [(0, "ema"), (0, "raw"), (2, "ema"), (2, "raw"),
                                                             (4, "ema"), (4, "raw")]
    assert all("elapsed" not in h for h in history)
    best = torch.load(tmp_path / "run/best.pt", weights_only=False)
    winner = min(history, key=lambda h: h["composite"])
    assert best["state"]["candidate_variant"] == winner["variant"] == summary["best"]["variant"]
    if winner["variant"] == "raw" and winner["step"] == 4:
        for name in cp.GENERATORS:
            equal(best["models"][name], last["models"][name])
    elif winner["variant"] == "ema" and winner["step"] == 4:
        for name in cp.GENERATORS:
            equal(best["models"][name], last["ema"]["models"][name])


def test_native_view_keeps_evaluator_scale_and_only_flips(tmp_path):
    data = dataset(tmp_path, monet=1, photo=1, size=64)
    path = cp.image_paths(data / "monet_jpg")[0]
    export = cp.load_image(path, 64, None)
    seen = set()
    for seed in range(20):
        view = cp.load_image(path, 64, random.Random(seed), "native")
        if torch.equal(view, export):
            seen.add("same")
        elif torch.equal(view, torch.flip(export, [2])):
            seen.add("flipped")
        else:
            raise AssertionError("native view must equal the export view or its mirror image")
    assert seen == {"same", "flipped"}


def test_config_rejects_unknown_variants_and_views():
    for bad in ({"eval_variants": "best"}, {"eval_variants": ""}, {"eval_variants": "ema,ema"}, {"train_view": "zoom"}):
        with pytest.raises(ValueError):
            cp.PushConfig(**bad)


def test_fast_ema_matches_reference_ema_bitwise():
    from lab1.cyclegan_ema import GeneratorEMA
    torch.manual_seed(3)
    models = cg.build_models({"base_channels": 2, "residual_blocks": 1}, "cpu")
    reference, fast = GeneratorEMA(models, 0.9), cp.FastGeneratorEMA(models, 0.9)
    for _ in range(5):
        with torch.no_grad():
            for name in cp.GENERATORS:
                for parameter in models[name].parameters():
                    parameter.add_(torch.randn_like(parameter))
        reference.update(models)
        fast.update(models)
    assert fast.updates == reference.updates == 5
    for name in cp.GENERATORS:
        equal(fast.models[name].state_dict(), reference.models[name].state_dict())
    assert set(fast.state_dict()) == set(reference.state_dict())


def test_stop_file_scores_checkpoints_and_resume_matches_continuous_training(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    push = recipe()
    full = cp.run(push, start, tmp_path / "full", "cpu", data, tmp_path, scorer=fake_scorer)
    stop = tmp_path / "STOP"
    stop.write_text("stop")
    stopped = cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, stop_file=stop)
    assert stopped["status"] == "partial" and stopped["completed_updates"] == 1
    history = torch.load(tmp_path / "split/last.pt", weights_only=False)["state"]["history"]
    assert [h["step"] for h in history] == [0, 1]
    stop.unlink()
    resumed = cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, resume=True,
                     stop_file=stop)
    assert resumed["status"] == full["status"] == "completed"
    one = torch.load(tmp_path / "full/last.pt", weights_only=False)
    two = torch.load(tmp_path / "split/last.pt", weights_only=False)
    for key in ("models", "optimizers", "replay_pools", "ema", "rng"):
        equal(one[key], two[key])


def test_batch_size_two_trains_resumes_exactly_and_batch_one_is_unchanged(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    push = recipe(batch_size=2)
    full = cp.run(push, start, tmp_path / "full", "cpu", data, tmp_path, scorer=fake_scorer)
    cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, stop_after=3)
    cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, resume=True)
    one = torch.load(tmp_path / "full/last.pt", weights_only=False)
    two = torch.load(tmp_path / "split/last.pt", weights_only=False)
    for key in ("models", "optimizers", "replay_pools", "ema", "rng"):
        equal(one[key], two[key])
    assert full["completed_updates"] == 6
    single = torch.load(cp.run(recipe(), start, tmp_path / "b1", "cpu", data, tmp_path, scorer=fake_scorer)
                        ["best_checkpoint"], weights_only=False)
    assert single["push_config"]["batch_size"] == 1


def test_slow_ema_is_scored_checkpointed_and_resumes_exactly(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    push = recipe(eval_variants="ema,ema_slow,raw", ema_slow_beta=0.99)
    cp.run(push, start, tmp_path / "full", "cpu", data, tmp_path, scorer=fake_scorer)
    cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, stop_after=3)
    cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, resume=True)
    one = torch.load(tmp_path / "full/last.pt", weights_only=False)
    two = torch.load(tmp_path / "split/last.pt", weights_only=False)
    for key in ("models", "optimizers", "replay_pools", "ema", "ema_slow", "rng"):
        equal(one[key], two[key])
    assert one["ema_slow"]["beta"] == 0.99 and one["ema_slow"]["updates"] == 6
    assert [h["variant"] for h in one["state"]["history"][:3]] == ["ema", "ema_slow", "raw"]
    for bad in ({"eval_variants": "ema_slow"}, {"ema_slow_beta": 1.0}, {"ema_slow_beta": -0.1}):
        with pytest.raises(ValueError):
            cp.PushConfig(**bad)


def test_lowpass_cycle_loss_blends_full_and_coarse_terms():
    torch.manual_seed(0)
    original, reconstruction = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    full = torch.nn.functional.l1_loss(reconstruction, original)
    assert torch.equal(cp.cycle_l1(reconstruction, original), full)
    coarse = torch.nn.functional.l1_loss(torch.nn.functional.avg_pool2d(reconstruction, 4),
                                         torch.nn.functional.avg_pool2d(original, 4))
    assert torch.allclose(cp.cycle_l1(reconstruction, original, 4, 0.25), 0.25 * full + 0.75 * coarse)
    # Pure high-frequency changes cost nothing in the coarse term.
    checker = (torch.arange(16).view(1, 1, 16, 1) + torch.arange(16).view(1, 1, 1, 16)) % 2 * 0.2 - 0.1
    assert float(cp.cycle_l1(original + checker, original, 4, 0.0)) < 1e-6
    for bad in ({"cycle_lowpass": 3}, {"cycle_detail": 1.5}, {"cycle_detail": 0.5}):
        with pytest.raises(ValueError):
            cp.PushConfig(**bad)
    cp.PushConfig(cycle_lowpass=4, cycle_detail=0.2)


def test_patch_nce_is_finite_and_trains_generator_and_heads():
    torch.manual_seed(0)
    generator = cg.Generator(4, 2)
    layers = cp.parse_layers("0,4,8,12")
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    channels = [f.shape[1] for f in cp.encoder_features(generator, x, layers)]
    heads = cp.PatchHeads(channels, width=16)
    loss = cp.patch_nce(heads, cp.encoder_features(generator, generator(x), layers),
                        cp.encoder_features(generator, x, layers), patches=32, tau=0.07)
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in generator.parameters())
    assert all(p.grad is not None for p in heads.parameters())
    # Identical query and key features give a lower loss than unrelated ones.
    with torch.no_grad():
        same = cp.patch_nce(heads, cp.encoder_features(generator, x, layers),
                            cp.encoder_features(generator, x, layers), 32, 0.07)
        other = cp.patch_nce(heads, cp.encoder_features(generator, torch.rand_like(x), layers),
                             cp.encoder_features(generator, x, layers), 32, 0.07)
    assert float(same) < float(other)


def test_nce_training_resumes_exactly_with_heads_checkpointed(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    push = recipe(nce_weight=1.0, nce_idt_weight=0.5, nce_layers="0,4,7", nce_patches=8)
    cp.run(push, start, tmp_path / "full", "cpu", data, tmp_path, scorer=fake_scorer)
    cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, stop_after=3)
    cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, resume=True)
    one = torch.load(tmp_path / "full/last.pt", weights_only=False)
    two = torch.load(tmp_path / "split/last.pt", weights_only=False)
    for key in ("models", "optimizers", "replay_pools", "ema", "rng", "nce_heads"):
        equal(one[key], two[key])
    assert set(one["nce_heads"]) == set(cp.GENERATORS)
    log = [json.loads(line) for line in (tmp_path / "full/training_log.jsonl").read_text().splitlines()
           if line.startswith('{"step"')]
    assert all(math.isfinite(r["nce_G_photo_to_monet"]) and "nce_idt_G_monet_to_photo" in r for r in log)
    for bad in ({"nce_idt_weight": 1.0}, {"nce_weight": -1.0}, {"nce_weight": 1.0, "nce_tau": 0.0},
                {"nce_weight": 1.0, "nce_layers": "-1"}, {"nce_weight": 1.0, "nce_patches": 1}):
        with pytest.raises(ValueError):
            cp.PushConfig(**bad)


def test_warm_start_from_this_trainers_checkpoint_into_more_or_fewer_scales(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    cp.run(recipe(steps=1, eval_every=1, extra_scales=0, d_warmup_steps=0), start, tmp_path / "one", "cpu", data,
           tmp_path, scorer=fake_scorer)
    single = tmp_path / "one/best.pt"
    saved = torch.load(single, weights_only=False)["models"]["D_monet"]
    assert any(k.startswith("scales.0.") for k in saved) and not any(k.startswith("scales.1.") for k in saved)
    config = torch.load(single, weights_only=False)["config"]
    gens, discs = cp.build(config, recipe(extra_scales=2), torch.load(single, weights_only=False)["models"], "cpu")
    assert len(discs["D_monet"].scales) == 3
    for key, value in discs["D_monet"].scales[0].state_dict().items():
        assert torch.equal(value, saved["scales.0." + key])
    three = {name: discs[name].state_dict() for name in cp.DISCRIMINATORS}
    gens, fewer = cp.build(config, recipe(extra_scales=1), {**{g: gens[g].state_dict() for g in cp.GENERATORS}, **three},
                           "cpu")
    assert len(fewer["D_monet"].scales) == 2
    for index in (0, 1):
        for key, value in fewer["D_monet"].scales[index].state_dict().items():
            assert torch.equal(value, three["D_monet"][f"scales.{index}.{key}"])


def test_report_metrics_kid_cosines_and_memorisation_check():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 16))
    same, shifted = rng.normal(size=(200, 16)), rng.normal(0.8, 1.0, size=(200, 16))
    assert abs(cp.kid(real, same)) < cp.kid(real, shifted)
    assert np.allclose(cp.paired_cosine(real, 3 * real), 1.0)
    copies = real[:5] + 1e-6
    assert np.allclose(cp.nearest_cosine(copies, real), 1.0)
    loo = cp.nearest_cosine(real, real, exclude_self=True)
    assert (loo < 1.0).all() and loo.shape == (200,)


def test_r1_penalty_is_zero_for_a_constant_discriminator_and_positive_otherwise():
    torch.manual_seed(0)
    real = torch.rand(2, 3, 64, 64) * 2 - 1
    discriminator = cp.MultiScaleDiscriminator(cg.PatchDiscriminator(4), 1, 4)
    penalty = cp.r1_penalty(discriminator, real)
    assert torch.isfinite(penalty) and float(penalty) > 0
    penalty.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in discriminator.parameters())

    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return [x.mean(dim=(1, 2, 3), keepdim=True) * 0 + self.bias]
    assert float(cp.r1_penalty(Constant(), real)) == 0.0


def test_r1_and_instance_noise_train_and_resume_exactly(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    push = recipe(r1_gamma=1.0, r1_every=2, d_noise_start=0.2, d_noise_end=0.0)
    cp.run(push, start, tmp_path / "full", "cpu", data, tmp_path, scorer=fake_scorer)
    cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, stop_after=3)
    cp.run(push, start, tmp_path / "split", "cpu", data, tmp_path, scorer=fake_scorer, resume=True)
    one = torch.load(tmp_path / "full/last.pt", weights_only=False)
    two = torch.load(tmp_path / "split/last.pt", weights_only=False)
    for key in ("models", "optimizers", "replay_pools", "ema", "rng"):
        equal(one[key], two[key])
    log = [json.loads(line) for line in (tmp_path / "full/training_log.jsonl").read_text().splitlines()
           if line.startswith('{"step"')]
    assert any("r1_monet" in r for r in log) and all(math.isfinite(r.get("r1_monet", 0.0)) for r in log)
    for bad in ({"r1_gamma": -1.0}, {"r1_every": 0}, {"d_noise_start": -0.1}):
        with pytest.raises(ValueError):
            cp.PushConfig(**bad)


def test_per_direction_cycle_scales_change_training_and_must_stay_positive(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    base = cp.run(recipe(steps=2, eval_every=2), start, tmp_path / "base", "cpu", data, tmp_path, scorer=fake_scorer)
    cp.run(recipe(steps=2, eval_every=2, cycle_monet_scale=0.4), start, tmp_path / "scaled", "cpu", data, tmp_path,
           scorer=fake_scorer)
    one = torch.load(tmp_path / "base/last.pt", weights_only=False)["models"]["G_monet_to_photo"]
    two = torch.load(tmp_path / "scaled/last.pt", weights_only=False)["models"]["G_monet_to_photo"]
    assert any(not torch.equal(one[k], two[k]) for k in one)
    assert base["status"] == "completed"
    for bad in ({"cycle_monet_scale": 0.0}, {"cycle_photo_scale": -1.0}):
        with pytest.raises(ValueError):
            cp.PushConfig(**bad)


def test_background_prefetch_gives_exactly_the_same_training_as_synchronous_loading(tmp_path):
    data, start = dataset(tmp_path), init_checkpoint(tmp_path)
    cp.run(recipe(prefetch=0), start, tmp_path / "sync", "cpu", data, tmp_path, scorer=fake_scorer)
    cp.run(recipe(prefetch=3), start, tmp_path / "ahead", "cpu", data, tmp_path, scorer=fake_scorer)
    one = torch.load(tmp_path / "sync/last.pt", weights_only=False)
    two = torch.load(tmp_path / "ahead/last.pt", weights_only=False)
    for key in ("models", "optimizers", "ema"):
        equal(one[key], two[key])
    batch_a = cp.sample_batch(cp.image_paths(data / "monet_jpg"), "monet", 7, 2, 32, 2342, "resize286")
    batch_b = cp.sample_batch(cp.image_paths(data / "monet_jpg"), "monet", 7, 2, 32, 2342, "resize286")
    assert torch.equal(batch_a, batch_b)
    assert not torch.equal(batch_a, cp.sample_batch(cp.image_paths(data / "monet_jpg"), "monet", 8, 2, 32, 2342,
                                                    "resize286"))


def test_learning_rate_warmup_ramps_then_follows_the_schedule():
    assert cp.lr_factor(0, 100, 0.5, warmup=10) == pytest.approx(0.1)
    assert cp.lr_factor(4, 100, 0.5, warmup=10) == pytest.approx(0.5)
    assert cp.lr_factor(9, 100, 0.5, warmup=10) == 1.0 and cp.lr_factor(30, 100, 0.5, warmup=10) == 1.0
    assert cp.lr_factor(75, 100, 0.5, warmup=10) == pytest.approx(cp.lr_factor(75, 100, 0.5))
    assert cp.lr_factor(3, 100, 0.5) == 1.0
    with pytest.raises(ValueError):
        cp.PushConfig(warmup_steps=-1)
