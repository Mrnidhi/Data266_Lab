"""CPU-only integration checks for the isolated online/EMA continuation."""
import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from lab1 import cyclegan as cg
from lab1 import cyclegan_ema as ce


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def recipe(**changes):
    return {"mode": "smoke", "seed": 2342, "image_size": 32, "base_channels": 2,
            "residual_blocks": 1, "batch_size": 1, "learning_rate": 5e-5, "betas": [.5, .999],
            "cycle_weight": 10., "identity_weight": 2.5, "replay_size": 1, "epochs": 8,
            "constant_epochs": 4, "max_steps": None, "checkpoint_every": 3, "log_every": 100,
            "precision": "fp32", "replay_device": "cpu", "allow_synthetic": True,
            "synthetic_images": 2, "cpu_threads": 2, "select_best": False,
            "evaluate_after_run": False, "ema_beta": .999, "validation_epochs": [4, 8],
            "monet_translation_ratio": 0., "evaluation": {"allow_metric_downloads": False},
            "data": {}, **changes}


def source(tmp_path, config):
    torch.manual_seed(17)
    models = cg.build_models(config, "cpu")
    _, provenance = cg.prepare_data(config)
    path = tmp_path / "source.pt"
    torch.save({"format_version": 1, "config": config,
                "state": {"global_step": 12, "manifest_fingerprint": provenance["manifest_fingerprint"]},
                "models": ce._cpu_states(models)}, path)
    return path


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


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
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def fake_evaluation(models, data, provenance, output_dir, device, split, options):
    # Deliberately consume all three RNGs and change modes: the observer must
    # isolate this from subsequent training, even if an evaluator is careless.
    random.random(); np.random.rand(); torch.rand(3)
    for model in models.values():
        model.eval()
    result = {"split": split, "class_results": provenance["class_results"],
              "data_kind": provenance["kind"], "manifest_fingerprint": provenance["manifest_fingerprint"], "directions": {}}
    for direction in cg.DIRECTIONS:
        src, target = direction.split("_to_")
        score = .01 + float(next(models["G_" + direction].parameters()).detach().abs().mean())
        result["directions"][direction] = {
            "source_count": len(data[f"val_{src}"]), "generated_count": len(data[f"val_{src}"]),
            "real_reference_count": len(data[f"val_{target}"]), "metrics": {"kid": {"status": "computed", "value": score}}}
    return result


def test_ema_math_fp32_initialization_buffers_and_no_rng_consumption():
    models = {name: nn.Linear(1, 1, bias=False).half() for name in ce.GENERATORS}
    for index, model in enumerate(models.values()):
        model.weight.data.fill_(1 + 2 * index)
        model.register_buffer("counter", torch.tensor(1))
    before = cg.rng_state()
    averaged = ce.GeneratorEMA(models, .5)
    equal(before, cg.rng_state())
    for index, model in enumerate(models.values()):
        model.weight.data.fill_(2 + 3 * index)
        model.counter.fill_(7)
    averaged.update(models)
    equal(before, cg.rng_state())
    assert averaged.updates == 1
    for name, value in zip(ce.GENERATORS, (1.5, 4.)):
        assert averaged.models[name].weight.dtype == torch.float32
        assert averaged.models[name].weight.item() == value
        assert averaged.models[name].counter.item() == 7
        assert not averaged.models[name].weight.requires_grad
    restored = ce.GeneratorEMA(models, .5)
    restored.load_state_dict(averaged.state_dict())
    equal(restored.state_dict(), averaged.state_dict())


def test_online_training_matches_existing_loop_exactly(tmp_path, monkeypatch):
    config = recipe(max_steps=3)
    start = source(tmp_path, config)
    monkeypatch.setattr(cg, "evaluate_models", fake_evaluation)
    ce.run(config, tmp_path / "ema", "cpu", warm_start=start)
    cg.run(config, tmp_path / "legacy", "cpu", warm_start=start)
    new, old = load(tmp_path / "ema/last.pt"), load(tmp_path / "legacy/last.pt")
    for key in ("models", "optimizers", "schedulers", "replay_pools", "rng"):
        equal(new[key], old[key])
    assert new["ema"]["updates"] == 3
    assert new["inference_only"] is False


def test_resume_matches_continuous_training_including_ema_rng_and_candidates(tmp_path, monkeypatch):
    config = recipe()
    start = source(tmp_path, config)
    monkeypatch.setattr(cg, "evaluate_models", fake_evaluation)
    ce.run(config, tmp_path / "continuous", "cpu", warm_start=start)
    partial = ce.run(dict(config, max_steps=7), tmp_path / "resumed", "cpu", warm_start=start)
    assert partial["status"] == "partial" and partial["completed_updates"] == 7
    resumed = ce.run(config, tmp_path / "resumed", "cpu", resume=tmp_path / "resumed/last.pt")
    assert resumed["status"] == "completed" and resumed["completed_epoch_fraction"] == 8
    continuous, restored = load(tmp_path / "continuous/last.pt"), load(tmp_path / "resumed/last.pt")
    for key in ("models", "optimizers", "schedulers", "replay_pools", "rng", "ema", "config"):
        equal(continuous[key], restored[key])
    assert {(r["step"], r["candidate_variant"]) for r in resumed["validation_history"]} == {
        (0, "raw"), (8, "raw"), (8, "ema"), (16, "raw"), (16, "ema")}
    for step in (8, 16):
        for variant in ("raw", "ema"):
            one = load(tmp_path / "continuous/candidates" / f"step_{step:07d}/{variant}.pt")
            two = load(tmp_path / "resumed/candidates" / f"step_{step:07d}/{variant}.pt")
            equal(one["models"], two["models"])


def test_candidate_export_contains_both_ema_generators_and_raw_discriminators(tmp_path, monkeypatch):
    config = recipe(max_steps=16)
    monkeypatch.setattr(cg, "evaluate_models", fake_evaluation)
    ce.run(config, tmp_path / "run", "cpu", warm_start=source(tmp_path, config))
    last = load(tmp_path / "run/last.pt")
    candidate = load(tmp_path / "run/candidates/step_0000016/ema.pt")
    assert candidate["format_version"] == 1 and candidate["inference_only"] is True
    assert not {"optimizers", "schedulers", "replay_pools", "rng"}.intersection(candidate)
    assert candidate["state"]["candidate_variant"] == "ema" and candidate["state"]["ema_updates"] == 16
    for name in ce.GENERATORS:
        equal(candidate["models"][name], last["ema"]["models"][name])
        assert any(not torch.equal(tensor, last["models"][name][key]) for key, tensor in candidate["models"][name].items())
    for name in ("D_photo", "D_monet"):
        equal(candidate["models"][name], last["models"][name])
    receipt = ce._read(tmp_path / "run/validation/step_0000016/ema/receipt.json")
    assert receipt["checkpoint_sha256"] == ce._sha256(tmp_path / "run/candidates/step_0000016/ema.pt")
    assert receipt["metrics_sha256"] == ce._sha256(tmp_path / "run/validation/step_0000016/ema/metrics.json")


@pytest.mark.parametrize("variant", ["raw", "ema"])
def test_validation_failure_preserves_rng_modes_and_online_weights(tmp_path, monkeypatch, variant):
    config = recipe()
    start = source(tmp_path, config)
    data, provenance = cg.prepare_data(config)
    models = cg.build_models(config, "cpu")
    averaged = ce.GeneratorEMA(models)
    averaged.update(models)
    state = {"global_step": 1, "manifest_fingerprint": provenance["manifest_fingerprint"],
             "initialization": {"checkpoint": str(start), "checkpoint_sha256": ce._sha256(start)}}
    before, weights = cg.rng_state(), ce._cpu_states(models)
    def fails(*args):
        fake_evaluation(*args)
        raise RuntimeError("intentional metric failure")
    monkeypatch.setattr(cg, "evaluate_models", fails)
    with pytest.raises(RuntimeError, match="intentional"):
        ce._validate_variant(models, averaged, data, provenance, tmp_path / "run", "cpu", config, state, variant)
    equal(before, cg.rng_state())
    equal(weights, ce._cpu_states(models))
    assert all(model.training for model in models.values())
    assert all(not model.training for model in averaged.models.values())
    assert not (tmp_path / "run/validation/step_0000001" / variant).exists()
    assert list((tmp_path / "run/validation/step_0000001").glob(f".{variant}-attempt-*"))


def test_interruption_between_raw_and_ema_validation_resumes_without_overwrite(tmp_path, monkeypatch):
    config = recipe()
    start = source(tmp_path, config)
    calls = []
    def fail_first_ema(*args):
        calls.append(args[3].name)
        if args[3].name.startswith(".ema-"):
            raise RuntimeError("interrupted EMA evaluation")
        return fake_evaluation(*args)
    monkeypatch.setattr(cg, "evaluate_models", fail_first_ema)
    with pytest.raises(RuntimeError, match="interrupted"):
        ce.run(config, tmp_path / "run", "cpu", warm_start=start)
    saved = load(tmp_path / "run/last.pt")
    assert saved["state"]["pending_validation"] == {"step": 8, "variants": ["ema"]}
    raw_path = tmp_path / "run/validation/step_0000008/raw/receipt.json"
    raw_hash = ce._sha256(raw_path)
    monkeypatch.setattr(cg, "evaluate_models", fake_evaluation)
    summary = ce.run(config, tmp_path / "run", "cpu", resume=tmp_path / "run/last.pt")
    assert summary["status"] == "completed" and ce._sha256(raw_path) == raw_hash
    assert len(summary["validation_history"]) == 5
    assert len(list((tmp_path / "run/validation/step_0000008").glob(".ema-attempt-*"))) == 1


def test_resume_rejects_recipe_source_log_and_inference_checkpoint_changes(tmp_path, monkeypatch):
    config = recipe(max_steps=1)
    start = source(tmp_path, config)
    monkeypatch.setattr(cg, "evaluate_models", fake_evaluation)
    ce.run(config, tmp_path / "run", "cpu", warm_start=start)
    last = tmp_path / "run/last.pt"
    with pytest.raises(ValueError, match="immutable recipe"):
        ce.run(recipe(max_steps=2, learning_rate=2e-5), tmp_path / "run", "cpu", resume=last)
    source_bytes = start.read_bytes(); start.write_bytes(source_bytes + b"changed")
    with pytest.raises(ValueError, match="Frozen source changed"):
        ce.run(recipe(max_steps=2), tmp_path / "run", "cpu", resume=last)
    start.write_bytes(source_bytes)
    log = tmp_path / "run/training_log.jsonl"; original = log.read_bytes()
    with log.open("a") as stream:
        stream.write('{"step":2}\n')
    with pytest.raises(ValueError, match="log-ahead"):
        ce.run(recipe(max_steps=2), tmp_path / "run", "cpu", resume=last)
    log.write_bytes(original)
    payload = load(last); payload["inference_only"] = True; torch.save(payload, last)
    with pytest.raises(ValueError, match="resumable online checkpoint"):
        ce.run(recipe(max_steps=2), tmp_path / "run", "cpu", resume=last)


def test_zero_wall_budget_saves_baseline_pending_and_resume_works(tmp_path, monkeypatch):
    config = recipe(max_steps=1)
    start = source(tmp_path, config)
    monkeypatch.setattr(cg, "evaluate_models", fake_evaluation)
    summary = ce.run(config, tmp_path / "run", "cpu", warm_start=start, wall_budget_seconds=0)
    assert summary["status"] == "partial" and summary["stop_reason"] == "budget_exhausted"
    assert summary["completed_updates"] == 0 and summary["pending_validation"] == {"step": 0, "variants": ["raw"]}
    resumed = ce.run(config, tmp_path / "run", "cpu", resume=tmp_path / "run/last.pt")
    assert resumed["completed_updates"] == 1 and resumed["ema_updates"] == 1 and resumed["pending_validation"] is None


def test_nonfinite_update_never_updates_online_or_ema(tmp_path, monkeypatch):
    config = recipe(max_steps=1)
    start = source(tmp_path, config)
    original = cg.generator_losses
    def nonfinite(*args, **kwargs):
        losses = original(*args, **kwargs)
        losses["generator_total"] = losses["generator_total"] * float("nan")
        return losses
    monkeypatch.setattr(cg, "evaluate_models", fake_evaluation)
    monkeypatch.setattr(cg, "generator_losses", nonfinite)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        ce.run(config, tmp_path / "run", "cpu", warm_start=start)
    saved = load(tmp_path / "run/last.pt")
    assert saved["state"]["global_step"] == saved["ema"]["updates"] == 0
    equal(saved["models"], load(start)["models"])
    assert ce._read(tmp_path / "run/training_failure.json")["nan_events"] == 1
