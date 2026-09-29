"""Control-flow/evidence checks for tuning; tiny fixtures are never lab results."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location("tune_cyclegan", Path(__file__).resolve().parents[1] / "scripts/tune_cyclegan.py")
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)


@pytest.fixture
def experiment(tmp_path):
    config = {"mode": "full", "seed": 2342, "image_size": 256, "base_channels": 64,
              "residual_blocks": 9, "batch_size": 1, "learning_rate": .0001,
              "betas": [.5, .999], "cycle_weight": 10., "identity_weight": 5.,
              "epochs": 30, "constant_epochs": 15, "replay_size": 50,
              "precision": "bf16", "replay_device": "device", "max_steps": None,
              "data": {"monet_dir": "fixture/monet", "photo_dir": "fixture/photo", "manifest_dir": "fixture/manifests"},
              "evaluation": {"seed": 2342, "kid_subsets": 50, "max_images": None}}
    source = tmp_path / "source.pt"
    torch.save({"format_version": 1, "config": config,
                "state": {"global_step": 168720, "manifest_fingerprint": "a" * 64},
                "models": {"fixture": torch.ones(1)}}, source)
    config_file = tmp_path / "config.json"
    tune.write_json(config_file, {"full": config})
    output = tmp_path / "experiment"
    protocol = tune.prepare(output, source, config_file)
    return output, source, config_file, protocol


def fake_train(config, run, device, resume=None, warm_start=None):
    """Write actual-format metadata without claiming a GPU/model experiment."""
    step = config.get("max_steps") or 20
    tune.write_json(run / "resolved_config.json", config)
    source_hash = tune.digest(warm_start) if warm_start else tune.read_json(run / "initialization.json")["checkpoint_sha256"]
    initialization = {"checkpoint_sha256": source_hash}
    tune.write_json(run / "initialization.json", initialization)
    torch.save({"config": config, "state": {"global_step": step, "initialization": initialization}}, run / "last.pt")
    print(f"fixture step={step}/20")
    return {"completed_updates": step, "completed_epoch_fraction": step / 2, "steps_per_epoch": 2,
            "initialization": initialization}


@pytest.fixture
def fake_training(monkeypatch):
    from lab1 import cyclegan
    monkeypatch.setattr(tune, "resolve_device", lambda device: device)
    monkeypatch.setattr(tune, "environment", lambda: {"fixture_only": True})
    monkeypatch.setattr(cyclegan, "run", fake_train)
    return cyclegan


def metrics(fingerprint, value):
    return {"split": "val", "class_results": True, "data_kind": "explicit_class_manifests",
            "manifest_fingerprint": fingerprint,
            "directions": {direction: {"source_count": 2, "generated_count": 2, "real_reference_count": 2,
                "metrics": {"kid": {"status": "computed", "value": value}}} for direction in tune.DIRECTIONS}}


def completed_arm(output, source, protocol, arm, value=.015, step=20):
    config = tune.read_json(output / protocol["arms"][arm]["config"])
    run = output / protocol["arms"][arm]["run_dir"]
    run.mkdir(parents=True)
    initialization = {"checkpoint_sha256": tune.digest(source)}
    selection = {"metric": tune.METRIC, "validation_only": True, "step": step, "value": value}
    fingerprint = protocol["source_manifest_fingerprint"]
    manifest = {"manifest_fingerprint": fingerprint, "counts": {"train_photo": 2, "train_monet": 2,
                                                                "val_photo": 2, "val_monet": 2}}
    summary = {"completed_updates": 20, "completed_epoch_fraction": 10, "nan_events": 0,
               "initialization": initialization, "best_selection": selection}
    wrapper = {"task": "cyclegan", "mode": "full", "status": "completed", "summary": summary,
               "config": config, "source_sha256": {"fixture": "b" * 64}, "environment": {"fixture": True}}
    tune.write_json(run / "run_summary.json", wrapper)
    tune.write_json(run / "resolved_config.json", config)
    tune.write_json(run / "data_manifest.json", manifest)
    tune.write_json(run / "validation/step_0000000/metrics.json", metrics(fingerprint, .02))
    tune.write_json(run / "validation" / f"step_{step:07d}" / "metrics.json", metrics(fingerprint, value))
    torch.save({"config": config, "state": {"global_step": step, "manifest_fingerprint": fingerprint,
                    "initialization": initialization, "best_selection": selection, "best_validation_score": value}}, run / "best.pt")
    return run


def test_prepare_freezes_equal_arms_except_identity_and_preserves_original(experiment):
    output, source, config_file, protocol = experiment
    source_before = tune.digest(source)
    left = tune.read_json(output / "configs/identity5.json")
    right = tune.read_json(output / "configs/identity2p5.json")
    assert left["identity_weight"] == 5 and right["identity_weight"] == 2.5
    assert {k: v for k, v in left.items() if k != "identity_weight"} == {k: v for k, v in right.items() if k != "identity_weight"}
    assert (left["epochs"], left["constant_epochs"], left["learning_rate"], left["precision"]) == (10, 5, 5e-5, "bf16")
    assert left["select_best"] and not left["evaluate_after_run"]
    assert left["evaluation"]["max_images"] is None
    assert protocol["training"]["validation_epochs"] == [0, 2, 4, 6, 8, 10]
    assert tune.digest(source) == source_before
    assert tune.read_json(config_file)["full"]["epochs"] == 30
    with pytest.raises(FileExistsError, match="new, empty"):
        tune.prepare(output, source, config_file)


@pytest.mark.parametrize("which", ["source", "config", "protocol"])
def test_modified_frozen_inputs_are_rejected(experiment, which):
    output, source, _, _ = experiment
    path = {"source": source, "config": output / "configs/identity5.json", "protocol": output / "protocol.json"}[which]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="modified"):
        tune.load_protocol(output)


def test_source_relocation_requires_same_hash(experiment, tmp_path):
    output, source, _, _ = experiment
    copied = tmp_path / "relocated.pt"
    copied.write_bytes(source.read_bytes())
    assert tune.load_protocol(output, copied)[2] == copied
    copied.write_bytes(b"another model")
    with pytest.raises(ValueError, match="different file"):
        tune.load_protocol(output, copied)


def test_capped_run_is_partial_and_explicit_resume_keeps_schedule(experiment, fake_training):
    output, source, _, _ = experiment
    before = tune.digest(source)
    partial = tune.train_arm(output, "identity5", max_steps=3)
    run = output / "arms/identity5"
    assert partial["status"] == "partial"
    assert partial["config"]["epochs"] == 10
    assert partial["task"] == "cyclegan" and partial["environment"] and partial["source_sha256"]
    assert "fixture step=3/20" in (run / "RUN_LOG.txt").read_text()
    with pytest.raises(FileExistsError, match="explicit --resume"):
        tune.train_arm(output, "identity5")
    complete = tune.train_arm(output, "identity5", resume=run / "last.pt")
    assert complete["status"] == "completed" and complete["resumed"]
    assert complete["config"]["max_steps"] is None
    assert len(list((run / "provenance").glob("*.json"))) == 2
    assert "fixture step=3/20" in (run / "RUN_LOG.txt").read_text()
    assert "fixture step=20/20" in (run / "RUN_LOG.txt").read_text()
    assert tune.digest(source) == before
    with pytest.raises(FileExistsError, match="already complete"):
        tune.train_arm(output, "identity5", resume=run / "last.pt")


def test_runner_records_failure_without_claiming_completion(experiment, fake_training, monkeypatch):
    output = experiment[0]
    def fail(*args, **kwargs):
        raise FloatingPointError("fixture NaN")
    monkeypatch.setattr(fake_training, "run", fail)
    with pytest.raises(FloatingPointError):
        tune.train_arm(output, "identity5")
    run = output / "arms/identity5"
    assert tune.read_json(run / "run_summary.json")["status"] == "failed"
    assert "fixture NaN" in (run / "RUN_LOG.txt").read_text()


def test_select_ranks_complete_arms_using_validation_and_keeps_source(experiment):
    output, source, _, protocol = experiment
    before = tune.digest(source)
    completed_arm(output, source, protocol, "identity5", .019)
    completed_arm(output, source, protocol, "identity2p5", .015)
    result = tune.select(output)
    assert result["winner"]["arm"] == "identity2p5"
    assert result["mean_kid_improvement"] == pytest.approx(.005)
    assert result["source_checkpoint_sha256"] == before
    assert result["winner"]["checkpoint_sha256"] == tune.digest(result["winner"]["checkpoint"])
    assert not result["published"] and not result["test_evaluated"]
    assert tune.digest(source) == before
    assert not list(output.rglob("evaluation*"))


def test_step_zero_best_retains_original_source(experiment):
    output, source, _, protocol = experiment
    for arm in tune.ARMS:
        completed_arm(output, source, protocol, arm, .02, step=0)
    result = tune.select(output)
    assert result["winner"]["kind"] == "unchanged_source"
    assert result["winner"]["checkpoint"] == str(source)
    assert result["mean_kid_improvement"] == 0


@pytest.mark.parametrize("problem", ["partial", "test_split", "unavailable", "nan", "bool", "count", "wrong_checkpoint", "baseline_disagrees"])
def test_invalid_or_incomparable_evidence_cannot_select(experiment, problem):
    output, source, _, protocol = experiment
    runs = [completed_arm(output, source, protocol, arm) for arm in tune.ARMS]
    run = runs[1]
    metric_path = run / "validation/step_0000020/metrics.json"
    if problem == "partial":
        path = run / "run_summary.json"
        data = tune.read_json(path)
        data["summary"]["completed_updates"] = 19
        tune.write_json(path, data)
    elif problem == "wrong_checkpoint":
        saved = tune.load_checkpoint(run / "best.pt")
        saved["state"]["global_step"] = 12
        torch.save(saved, run / "best.pt")
    elif problem == "baseline_disagrees":
        tune.write_json(run / "validation/step_0000000/metrics.json", metrics(protocol["source_manifest_fingerprint"], .03))
    else:
        data = tune.read_json(metric_path)
        entry = data["directions"][tune.DIRECTIONS[0]]
        if problem == "test_split":
            data["split"] = "test"
        elif problem == "unavailable":
            entry["metrics"]["kid"]["status"] = "unavailable"
        elif problem == "nan":
            entry["metrics"]["kid"]["value"] = float("nan")
        elif problem == "bool":
            entry["metrics"]["kid"]["value"] = True
        elif problem == "count":
            entry["generated_count"] = 1
        metric_path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        tune.select(output)
    assert not (output / "selection.json").exists()
