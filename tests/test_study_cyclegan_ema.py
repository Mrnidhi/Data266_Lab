"""CPU-only receipt/selection tests; no image metrics or GPU experiments."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("ema_study", Path(__file__).resolve().parents[1] / "scripts/study_cyclegan_ema.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def networks():
    return {name: {"weight": torch.ones(1)} for name in ("G_photo_to_monet", "G_monet_to_photo", "D_photo", "D_monet")}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(study, "ROOT", root)
    for relative in study.SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen fixture source\n")
    config = {"mode": "full", "seed": 2342, "image_size": 32, "base_channels": 2,
              "residual_blocks": 1, "batch_size": 1, "epochs": 10,
              "data": {"monet_dir": "monet", "photo_dir": "photo", "manifest_dir": "manifests"}}
    source = root / "source.pt"
    torch.save({"format_version": 1, "config": config, "models": networks(),
                "state": {"global_step": 100, "manifest_fingerprint": "a" * 64}}, source)
    config_path = root / "config.json"
    study.write_json(config_path, {"full": config})
    output = root / "study"
    protocol = study.prepare(output, source, config_path, study.digest(source))
    return output, source, protocol


def metrics(protocol, values):
    return {"split": "val", "class_results": True, "data_kind": "explicit_class_manifests",
            "manifest_fingerprint": protocol["source_manifest_fingerprint"],
            "directions": {direction: {"source_count": 2, "generated_count": 2, "real_reference_count": 2,
                "metrics": {"kid": {"status": "computed", "value": value}}}
                for direction, value in zip(study.DIRECTIONS, values)}}


def validation_evidence(directory, checkpoint, protocol, step, variant, values):
    metric = dict(metrics(protocol, values), checkpoint_sha256=study.digest(checkpoint), checkpoint_step=step,
                  candidate_variant=variant, ema_beta=.999, ema_updates=step)
    study.write_json(directory / "metrics.json", metric)
    receipt = {"status": "complete", "trainer": "cyclegan_ema_v1", "checkpoint_sha256": study.digest(checkpoint),
        "metrics_sha256": study.digest(directory / "metrics.json"), "candidate_variant": variant,
        "step": step, "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
        "manifest_fingerprint": protocol["source_manifest_fingerprint"],
        "checkpoint": str(checkpoint), "metrics_path": str(directory / "metrics.json"),
        "directional_kid": dict(zip(study.DIRECTIONS, values)), "mean_validation_kid": sum(values) / 2}
    study.write_json(directory / "receipt.json", receipt)
    return receipt


def completed_arm(output, protocol, arm, values=(.018, .018)):
    study.budget_receipt(output, protocol)
    run = output / protocol["arms"][arm]["run_dir"]
    config = study.read_json(output / protocol["arms"][arm]["config"])
    init = {"checkpoint_sha256": protocol["source_checkpoint_sha256"]}
    summary = {"status": "completed", "pending_validation": None, "completed_updates": 16,
               "completed_epoch_fraction": 8, "steps_per_epoch": 2, "nan_events": 0,
               "trainer": "cyclegan_ema_v1", "ema_beta": .999, "ema_updates": 16,
               "initialization": init}
    wrapper = {"status": "completed", "task": "cyclegan_ema", "arm": arm, "config": config,
               "source_sha256": protocol["source_code_hashes"], "protocol_sha256": study.digest(output / "protocol.json"),
               "budget_sha256": study.digest(output / "budget.json"),
               "warm_start_source_sha256": protocol["source_checkpoint_sha256"], "environment": {"fixture": True},
               "started_utc": "2026-10-02T00:00:00+00:00", "ended_utc": "2026-10-02T00:00:01+00:00",
               "elapsed_seconds": 1, "summary": summary}
    study.write_json(run / "run_summary.json", wrapper)
    study.write_json(run / "provenance/final.json", wrapper)
    study.write_json(run / "resolved_config.json", config)
    study.write_json(run / "data_manifest.json", {"manifest_fingerprint": protocol["source_manifest_fingerprint"],
        "counts": {"train_photo": 2, "train_monet": 2, "val_photo": 2, "val_monet": 2}})
    with (run / "training_log.jsonl").open("w") as f:
        for step in range(1, 17):
            f.write(json.dumps({"step": step, "nan_events": 0, "losses": {"generator_total": .1},
                               "gradient_norms": {"G_photo_to_monet": 1.0}}) + "\n")
    receipts = [validation_evidence(run / "validation/step_0000000/raw", study.resolve_path(protocol["source_checkpoint"]),
                                   protocol, 0, "raw", (.02, .02))]
    for step in (8, 16):
        for variant in study.VARIANTS:
            path = run / "candidates" / f"step_{step:07d}" / f"{variant}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format_version": 1, "trainer": "cyclegan_ema_v1", "inference_only": True, "config": config, "models": networks(),
                "state": {"global_step": step, "candidate_variant": variant, "manifest_fingerprint": protocol["source_manifest_fingerprint"],
                          "initialization": init, "ema_beta": .999, "ema_updates": step}}, path)
            directory = run / "validation" / f"step_{step:07d}" / variant
            receipts.append(validation_evidence(directory, path, protocol, step, variant, values))
    summary["validation_history"] = receipts
    study.write_json(run / "run_summary.json", wrapper)
    study.write_json(run / "provenance/final.json", wrapper)
    study.write_json(run / "trainer_summary.json", summary)
    return run


def test_prepare_freezes_only_lr_difference_and_preserves_source(experiment):
    output, source, p = experiment
    left, right = [study.read_json(output / p["arms"][arm]["config"]) for arm in study.ARMS]
    assert left["learning_rate"] == 5e-5 and right["learning_rate"] == 2e-5
    assert {k: v for k, v in left.items() if k != "learning_rate"} == {k: v for k, v in right.items() if k != "learning_rate"}
    assert (left["epochs"], left["constant_epochs"], left["ema_beta"], left["identity_weight"], left["cycle_weight"]) == (8, 4, .999, 2.5, 10)
    assert left["validation_epochs"] == [4, 8] and left["monet_translation_ratio"] == 0
    assert study.digest(source) == p["source_checkpoint_sha256"]
    assert p["selection"]["max_candidates"] == 9 and p["budget"]["total_seconds"] == 21600


@pytest.mark.parametrize("target", ["protocol", "config", "source", "code", "snapshot"])
def test_modified_frozen_inputs_rejected(experiment, target):
    output, source, p = experiment
    path = {"protocol": output / "protocol.json", "config": output / "configs/lr5e5.json",
            "source": source, "code": study.ROOT / study.SOURCES[0],
            "snapshot": output / "source_snapshot" / study.SOURCES[0]}[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="modified"):
        study.load_protocol(output)


def test_guardrail_excludes_mean_improvement_with_directional_regression(experiment):
    output, _, p = experiment
    completed_arm(output, p, "lr5e5", (.001, .021))
    completed_arm(output, p, "lr2e5", (.019, .019))
    selected = study.select(output)
    assert len(selected["all_candidates"]) == 9
    assert selected["winner"]["arm"] == "lr2e5"
    excluded = [r for r in selected["all_candidates"] if r["arm"] == "lr5e5"]
    assert all(not r["eligible"] and r["exclusion_reason"] for r in excluded)
    assert selected["winner"]["variant"] == "raw" and selected["winner"]["step"] == 8
    assert not selected["test_evaluated"] and not selected["class_evaluated"]


def test_exact_tie_keeps_original_source(experiment):
    output, source, p = experiment
    for arm in study.ARMS:
        completed_arm(output, p, arm, (.02, .02))
    selected = study.select(output)
    assert selected["winner"]["kind"] == "unchanged_source"
    assert (output / selected["winner"]["checkpoint"]).resolve() == source


@pytest.mark.parametrize("problem", ["partial", "pending", "missing_end", "stale_provenance", "nan_summary", "gap_log", "nan_log", "stale_metric", "wrong_variant", "nan_tensor", "baseline_receipt", "incomplete_receipt", "stale_history", "ema_updates"])
def test_untrustworthy_completion_or_candidate_rejected(experiment, problem):
    output, _, p = experiment
    runs = {arm: completed_arm(output, p, arm) for arm in study.ARMS}
    run = runs["lr5e5"]
    wrapper = study.read_json(run / "run_summary.json")
    if problem in ("partial", "pending", "missing_end", "stale_provenance", "nan_summary"):
        if problem == "partial": wrapper["status"] = "partial"
        if problem == "pending": wrapper["summary"]["pending_validation"] = {"step": 16}
        if problem == "missing_end": wrapper.pop("ended_utc")
        if problem == "nan_summary": wrapper["summary"]["nan_events"] = 1
        if problem == "stale_provenance": wrapper["elapsed_seconds"] = 2
        study.write_json(run / "run_summary.json", wrapper)
        if problem != "stale_provenance": study.write_json(run / "provenance/final.json", wrapper)
    elif problem == "stale_history":
        wrapper["summary"]["validation_history"].pop()
        study.write_json(run / "run_summary.json", wrapper)
        study.write_json(run / "provenance/final.json", wrapper)
        study.write_json(run / "trainer_summary.json", wrapper["summary"])
    elif problem in ("baseline_receipt", "incomplete_receipt"):
        directory = run / ("validation/step_0000000/raw" if problem == "baseline_receipt" else "validation/step_0000008/ema")
        receipt = study.read_json(directory / "receipt.json")
        if problem == "baseline_receipt": receipt["checkpoint_sha256"] = "0" * 64
        else: receipt["status"] = "partial"
        study.write_json(directory / "receipt.json", receipt)
    elif problem in ("gap_log", "nan_log"):
        records = [json.loads(l) for l in (run / "training_log.jsonl").read_text().splitlines()]
        if problem == "gap_log": records[3]["step"] = 3
        else: records[3]["losses"]["generator_total"] = float("nan")
        (run / "training_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    else:
        directory = run / "validation/step_0000008/ema"
        if problem == "stale_metric":
            (directory / "metrics.json").write_bytes((directory / "metrics.json").read_bytes() + b" ")
        else:
            path = run / "candidates/step_0000008/ema.pt"
            saved = study.load_checkpoint(path)
            if problem == "wrong_variant": saved["state"]["candidate_variant"] = "raw"
            elif problem == "ema_updates": saved["state"]["ema_updates"] = 7
            else: saved["models"]["G_photo_to_monet"]["weight"].fill_(float("nan"))
            torch.save(saved, path)
            metric = study.read_json(directory / "metrics.json")
            metric["checkpoint_sha256"] = study.digest(path)
            study.write_json(directory / "metrics.json", metric)
            receipt = study.read_json(directory / "receipt.json")
            receipt["checkpoint_sha256"] = study.digest(path)
            receipt["metrics_sha256"] = study.digest(directory / "metrics.json")
            study.write_json(directory / "receipt.json", receipt)
    with pytest.raises(ValueError):
        study.select(output)
    assert not (output / "selection.json").exists()


@pytest.fixture
def fake_training(monkeypatch):
    from lab1 import cyclegan_ema
    monkeypatch.setattr(study, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(study, "environment", lambda: {"fixture": True})
    def fake(config, run, device, resume=None, warm_start=None, deadline_utc=None):
        step = config["max_steps"] or 16
        init = {"checkpoint_sha256": config["expected_source_sha256"]}
        study.write_json(run / "resolved_config.json", config)
        torch.save({"trainer": "cyclegan_ema_v1", "inference_only": False, "config": config,
                    "state": {"global_step": step, "initialization": init}}, run / "last.pt")
        assert deadline_utc
        return {"status": "completed" if step == 16 else "partial", "pending_validation": None,
                "completed_updates": step, "completed_epoch_fraction": step/2, "steps_per_epoch": 2,
                "nan_events": 0, "initialization": init}
    monkeypatch.setattr(cyclegan_ema, "run", fake)
    return cyclegan_ema


def test_benchmark_explicit_resume_keeps_frozen_schedule_and_budget(experiment, fake_training):
    output, _, _ = experiment
    partial = study.train_arm(output, "lr5e5", device="cpu", max_steps=2, wall_budget_seconds=60)
    assert partial["status"] == "partial" and partial["config"]["epochs"] == 8
    budget_hash = study.digest(output / "budget.json")
    with pytest.raises(FileExistsError, match="explicit"):
        study.train_arm(output, "lr5e5", device="cpu")
    completed = study.train_arm(output, "lr5e5", device="cpu", resume=output / "arms/lr5e5/last.pt")
    assert completed["status"] == "completed" and completed["config"]["max_steps"] is None
    assert study.digest(output / "budget.json") == budget_hash
    with pytest.raises(FileExistsError, match="already completed"):
        study.train_arm(output, "lr5e5", device="cpu", resume=output / "arms/lr5e5/last.pt")


def test_inference_candidate_refused_for_resume(experiment, fake_training):
    output, _, _ = experiment
    study.train_arm(output, "lr5e5", device="cpu", max_steps=2)
    path = output / "arms/lr5e5/last.pt"
    saved = study.load_checkpoint(path); saved["inference_only"] = True; torch.save(saved, path)
    with pytest.raises(ValueError, match="Resume checkpoint"):
        study.train_arm(output, "lr5e5", device="cpu", resume=path)


def test_full_update_count_with_pending_validation_remains_partial(experiment, fake_training, monkeypatch):
    original = fake_training.run
    def pending(*args, **kwargs):
        result = original(*args, **kwargs)
        result.update(status="partial", pending_validation={"step": 16})
        return result
    monkeypatch.setattr(fake_training, "run", pending)
    assert study.train_arm(experiment[0], "lr5e5", device="cpu")["status"] == "partial"


def test_lock_is_nonblocking(tmp_path):
    with study.runner_lock(tmp_path / "runner.lock"):
        with pytest.raises(RuntimeError, match="live runner"):
            with study.runner_lock(tmp_path / "runner.lock"):
                pytest.fail("Second runner acquired lock")


def test_budget_reader_verifies_clock_and_preserves_expired_finalization_access(experiment):
    from datetime import datetime, timedelta, timezone
    output, _, p = experiment
    study.budget_receipt(output, p)
    now = datetime.now(timezone.utc)
    budget = study.read_budget(output)
    budget.update(started_utc=(now-timedelta(hours=7)).isoformat(), deadline_utc=(now-timedelta(hours=1)).isoformat(),
                  training_deadline_utc=(now-timedelta(hours=1, minutes=15)).isoformat())
    study.write_json(output / "budget.json", budget)
    (output / "budget.sha256").write_text(study.digest(output / "budget.json"))
    assert study.read_budget(output) == budget
    with pytest.raises(TimeoutError):
        study.budget_receipt(output, p)
    budget["deadline_utc"] = now.isoformat()
    study.write_json(output / "budget.json", budget)
    (output / "budget.sha256").write_text(study.digest(output / "budget.json"))
    with pytest.raises(ValueError, match="timestamps"):
        study.read_budget(output)
