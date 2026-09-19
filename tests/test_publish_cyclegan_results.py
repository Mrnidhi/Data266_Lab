"""Publication accepts traceable saved evidence, never incomplete or invented results."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat
from PIL import Image
import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "publish_cyclegan_results", Path(__file__).resolve().parents[1] / "scripts/publish_cyclegan_results.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


@pytest.fixture
def actual_format_fixture(tmp_path):
    """Tiny fabricated test artifacts, never a trained lab result."""
    run, evaluation = tmp_path / "run", tmp_path / "evaluation"
    run.mkdir()
    evaluation.mkdir()
    config = {"mode": "full", "seed": 2342, "image_size": 256, "base_channels": 64, "residual_blocks": 9,
              "batch_size": 1, "learning_rate": .0001, "betas": [.5, .999], "cycle_weight": 10.,
              "identity_weight": 5., "epochs": 2, "constant_epochs": 1, "max_steps": None, "select_best": True,
              "replay_size": 50, "precision": "fp32"}
    entries = {f"{split}_{domain}": [{"relative_path": f"{split}-{i}.jpg",
                                    "sha256": hashlib.sha256(f"{split}/{domain}/{i}".encode()).hexdigest()}
                                   for i in range(2)]
               for split in ("train", "val", "test") for domain in ("photo", "monet")}
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    data = {"kind": "explicit_class_manifests", "class_results": True, "counts": {k: 2 for k in entries},
            "entries": entries, "manifest_fingerprint": fingerprint}
    selection = {"metric": "mean_validation_KID_both_directions", "validation_only": True, "step": 2, "value": .1}
    summary = {"mode": "full", "class_results": True, "data_kind": data["kind"], "steps_per_epoch": 2,
               "completed_updates": 4, "completed_epoch_fraction": 2, "training_seconds": 8., "session_wall_seconds": 10.,
               "training_source_images_per_second": 1., "peak_gpu_memory_bytes": 1024, "nan_events": 0,
               "parameter_counts": {name: 4 for name in publisher.NETWORKS}, "best_selection": selection,
               "throughput_definition": "Two source images per update; excludes loading and evaluation"}
    provenance = {"task": "cyclegan", "status": "completed", "mode": "full", "config": config,
                  "source_sha256": {"training.py": "a" * 64}, "summary": summary,
                  "environment": {"gpus": [{"name": "Fixture training GPU"}], "platform": "Fixture Linux", "python": "3.12"}}
    for name, value in (("run_summary.json", provenance), ("resolved_config.json", config), ("data_manifest.json", data)):
        publisher.write_json(run / name, value)
    (run / "RUN_LOG.txt").write_bytes(b"Unedited fixture console log\r\ncompleted\n")
    records = [{"step": step, "losses": {name: float(step) / 10 for name in publisher.LOSS_NAMES},
                "gradient_norms": {name: float(step) for name in publisher.NETWORKS}, "step_seconds": 2., "nan_events": 0}
               for step in range(1, 5)]
    (run / "training_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    checkpoint = run / "best.pt"
    torch.save({"format_version": 1, "config": config,
                "state": {"global_step": 2, "manifest_fingerprint": fingerprint,
                          "best_selection": selection, "best_validation_score": selection["value"]},
                "models": {name: {"fixture": torch.zeros(4)} for name in publisher.NETWORKS}}, checkpoint)
    metrics = {"split": "test", "class_results": True, "data_kind": data["kind"], "manifest_fingerprint": fingerprint,
               "checkpoint_sha256": publisher.digest(checkpoint), "checkpoint_step": 2, "directions": {}}
    for direction in publisher.DIRECTIONS:
        metrics["directions"][direction] = {"source_count": 2, "generated_count": 2, "real_reference_count": 2,
            "inference_seconds": .2, "inference_images_per_second": 10.,
            "metrics": {name: {"value": .1, "status": "computed"} for name in publisher.METRICS}}
        for folder in ("source", "generated", "cycle", "real_target"):
            destination = evaluation / direction / folder
            destination.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (8, 8), (i * 10, 20, 30)).save(destination / f"{i:06d}.png")
    metrics["directions"]["photo_to_monet"]["metrics"]["lpips_cycle"] = {
        "value": None, "status": "unavailable", "reason": "Fixture missing optional evaluator"}
    publisher.write_json(evaluation / "metrics.json", metrics)
    commands = tmp_path / "commands.json"
    publisher.write_json(commands, {"training": ["python -m lab1.run --task cyclegan --mode full --output runs/example --device cuda"],
                                    "evaluation": "python task3_gan/srinidhi/evaluate_local.py --checkpoint runs/example/best.pt --output-dir runs/evaluation"})
    return {"run_dir": run, "evaluation_dir": evaluation, "checkpoint": checkpoint, "commands_file": commands}


def modify(path, function):
    document = json.loads(path.read_text())
    function(document)
    publisher.write_json(path, document)


def test_dry_run_has_no_writes_and_preserves_missing_human_and_kaggle_values(actual_format_fixture, tmp_path):
    evidence = publisher.load_evidence(**actual_format_fixture)
    destination = tmp_path / "does-not-exist" / "publication"
    result = publisher.publish(evidence, destination)
    assert not destination.parent.exists() and not result["write"]
    rows = publisher.metric_rows(evidence)
    pending = [row for row in rows if row["scope"] in {"human_audit", "kaggle"}]
    assert pending and all(row["value"] == "" and row["status"] == "pending" for row in pending)
    lpips = next(row for row in rows if row["scope"] == "test" and row["direction"] == "photo_to_monet" and row["metric"] == "lpips_cycle")
    assert lpips["value"] == "" and "Fixture missing" in lpips["details"]
    gpu = next(row for row in rows if row["scope"] == "hardware" and row["metric"] == "gpu")
    assert gpu["value"] == "Fixture training GPU"
    assert all(row["checkpoint_sha256"] == evidence["checkpoint_sha256"] for row in rows)


@pytest.mark.parametrize("mutation, error", [
    (lambda x: x["summary"].update(completed_updates=3), "Incomplete training"),
    (lambda x: x["summary"].update(class_results=False), "completed full run"),
    (lambda x: x.update(status="failed"), "completed full run"),
])
def test_rejects_incomplete_or_synthetic_even_with_plausible_metrics(actual_format_fixture, mutation, error):
    modify(actual_format_fixture["run_dir"] / "run_summary.json", mutation)
    with pytest.raises(ValueError, match=error):
        publisher.load_evidence(**actual_format_fixture)


@pytest.mark.parametrize("mutation, error", [
    (lambda x: x.update(checkpoint_sha256="f" * 64), "checkpoint hash mismatch"),
    (lambda x: x.update(manifest_fingerprint="f" * 64), "frozen manifests"),
    (lambda x: x.update(split="val"), "Held-out test"),
    (lambda x: x["directions"]["photo_to_monet"].update(source_count=1, generated_count=1), "Incomplete evaluation"),
    (lambda x: x["directions"]["photo_to_monet"]["metrics"]["fid"].update(value=float("nan")), "Non-finite"),
])
def test_rejects_unbound_partial_or_nonfinite_evaluation(actual_format_fixture, mutation, error):
    path = actual_format_fixture["evaluation_dir"] / "metrics.json"
    document = json.loads(path.read_text())
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=error):
        publisher.load_evidence(**actual_format_fixture)


@pytest.mark.parametrize("mutation, error", [
    (lambda x: x["config"].update(precision="bf16"), "recipe.*precision"),
    (lambda x: x["config"].update(replay_size=10), "recipe.*replay_size"),
    (lambda x: x["state"]["best_selection"].update(value=.2), "validation-selected best"),
    (lambda x: x["state"].pop("best_selection"), "validation-selected best"),
    (lambda x: x["state"].update(best_validation_score=.2), "validation-selected best"),
])
def test_checkpoint_recipe_and_saved_selection_must_match_run(actual_format_fixture, mutation, error):
    checkpoint = actual_format_fixture["checkpoint"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mutation(saved)
    torch.save(saved, checkpoint)
    # Refresh the evaluation hash so rejection tests semantic provenance, not
    # merely the already-covered byte-hash mismatch.
    modify(actual_format_fixture["evaluation_dir"] / "metrics.json",
           lambda x: x.update(checkpoint_sha256=publisher.digest(checkpoint)))
    with pytest.raises(ValueError, match=error):
        publisher.load_evidence(**actual_format_fixture)


def test_legacy_checkpoint_precision_defaults_to_fp32(actual_format_fixture):
    checkpoint = actual_format_fixture["checkpoint"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["config"].pop("precision")
    torch.save(saved, checkpoint)
    modify(actual_format_fixture["evaluation_dir"] / "metrics.json",
           lambda x: x.update(checkpoint_sha256=publisher.digest(checkpoint)))
    evidence = publisher.load_evidence(**actual_format_fixture)
    assert evidence["config"]["precision"] == "fp32"


@pytest.mark.parametrize("changes, error", [
    ({"training_seconds": 7.}, "training time differs"),
    ({"training_source_images_per_second": .5}, "throughput differs"),
])
def test_training_timing_summary_must_match_retained_log_evidence(actual_format_fixture, changes, error):
    modify(actual_format_fixture["run_dir"] / "run_summary.json", lambda x: x["summary"].update(changes))
    with pytest.raises(ValueError, match=error):
        publisher.load_evidence(**actual_format_fixture)


def test_requires_full_resumed_history_and_detects_conflicting_records(actual_format_fixture, tmp_path):
    run = actual_format_fixture["run_dir"]
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    for name in ("data_manifest.json", "RUN_LOG.txt", "resolved_config.json"):
        (earlier / name).write_bytes((run / name).read_bytes())
    lines = (run / "training_log.jsonl").read_text().splitlines(keepends=True)
    (earlier / "training_log.jsonl").write_text("".join(lines[:2]))
    (run / "training_log.jsonl").write_text("".join(lines[2:]))
    with pytest.raises(ValueError, match="Incomplete training history"):
        publisher.load_evidence(**actual_format_fixture)
    evidence = publisher.load_evidence(**actual_format_fixture, history_run_dirs=[earlier])
    assert [row["step"] for row in evidence["logs"]] == [1, 2, 3, 4]
    conflicting = json.loads(lines[2])
    conflicting["losses"]["generator_total"] += 1
    with (earlier / "training_log.jsonl").open("a") as stream:
        stream.write(json.dumps(conflicting) + "\n")
    with pytest.raises(ValueError, match="Conflicting resumed"):
        publisher.load_evidence(**actual_format_fixture, history_run_dirs=[earlier])


def test_explicit_rollback_cutoff_preserves_old_logs_and_uses_resumed_updates(actual_format_fixture, tmp_path):
    run, earlier = actual_format_fixture["run_dir"], tmp_path / "before-interruption"
    earlier.mkdir()
    for name in ("data_manifest.json", "RUN_LOG.txt", "resolved_config.json"):
        (earlier / name).write_bytes((run / name).read_bytes())
    lines = (run / "training_log.jsonl").read_text().splitlines(keepends=True)
    # Update 3 was logged but not checkpointed. The completed resumed run repeats
    # it after update 2; its legitimate measured timing/loss differs.
    (earlier / "training_log.jsonl").write_text("".join(lines[:3]))
    repeated = json.loads(lines[2])
    repeated["step_seconds"] = 2.5
    repeated["losses"]["generator_total"] = .75
    (run / "training_log.jsonl").write_text(json.dumps(repeated) + "\n" + lines[3])
    original = (earlier / "training_log.jsonl").read_bytes()
    modify(run / "run_summary.json", lambda x: x.update(resumed=True))
    modify(run / "run_summary.json", lambda x: x["summary"].update(
        training_seconds=8.5, training_source_images_per_second=8 / 8.5))
    with pytest.raises(ValueError, match="Conflicting resumed"):
        publisher.load_evidence(**actual_format_fixture, history_run_dirs=[earlier])
    evidence = publisher.load_evidence(**actual_format_fixture, history_run_dirs=[earlier], history_cutoffs={earlier: 2})
    assert len(evidence["logs"]) == 4 and evidence["logs"][2] == repeated
    assert evidence["history_reconstruction"][0]["excluded_unsaved_tail_records"] == 1
    assert (earlier / "training_log.jsonl").read_bytes() == original
    destination = tmp_path / "recovered-publication"
    publisher.publish(evidence, destination, write=True)
    assert (destination / "outputs/full/raw_logs/run_00/training_log.jsonl").read_bytes() == original
    reconstruction = publisher.read_json(destination / "outputs/full/history_reconstruction.json")
    assert reconstruction[0]["explicit_saved_update_cutoff"] == 2
    assert reconstruction[0]["raw_log_sha256"] == hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match="explicit earlier"):
        publisher.load_evidence(**actual_format_fixture, history_run_dirs=[earlier], history_cutoffs={run: 2})
    modify(earlier / "resolved_config.json", lambda x: x.update(seed=999))
    with pytest.raises(ValueError, match="different training recipe"):
        publisher.load_evidence(**actual_format_fixture, history_run_dirs=[earlier], history_cutoffs={earlier: 2})


def test_data_manifest_tampering_and_missing_image_evidence_fail_closed(actual_format_fixture):
    path = actual_format_fixture["run_dir"] / "data_manifest.json"
    original = path.read_bytes()
    modify(path, lambda x: x["entries"]["test_photo"][0].update(sha256="e" * 64))
    with pytest.raises(ValueError, match="fingerprint"):
        publisher.load_evidence(**actual_format_fixture)
    path.write_bytes(original)
    (actual_format_fixture["evaluation_dir"] / "photo_to_monet/generated/000001.png").unlink()
    with pytest.raises(ValueError, match="Missing image evidence"):
        publisher.load_evidence(**actual_format_fixture)


def test_write_executes_saved_evidence_notebook_and_never_overwrites_existing_artifacts(actual_format_fixture, tmp_path):
    evidence = publisher.load_evidence(**actual_format_fixture)
    sources = {path: path.read_bytes() for path in actual_format_fixture["run_dir"].iterdir() if path.is_file()}
    destination = tmp_path / "publication"
    result = publisher.publish(evidence, destination, write=True)
    assert result["write"] and not result["all_lab_requirements_complete"]
    manifest = publisher.read_json(destination / "publication.json")
    for relative, info in manifest["files"].items():
        assert publisher.digest(destination / relative) == info["sha256"]
    for path, content in sources.items():
        assert path.read_bytes() == content
    assert (destination / "outputs/full/raw_logs/run_00/RUN_LOG.txt").read_bytes() == sources[actual_format_fixture["run_dir"] / "RUN_LOG.txt"]
    notebook = nbformat.read(destination / "src/cyclegan.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 5 and all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
    assert any("image/png" in output.get("data", {}) for cell in code for output in cell.outputs)
    assert "summarizes a saved completed training run" in notebook.cells[0].source
    assert publisher.read_json(destination / "notebook_execution.json")["execution_kind"] == "saved_artifact_presentation"
    assert (destination / "metrics_report.csv").read_bytes() == (destination / "full_metrics_report.csv").read_bytes()
    rows = list(csv.DictReader((destination / "metrics_report.csv").open()))
    assert any(row["scope"] == "hardware" and row["value"] == "Fixture training GPU" for row in rows)
    before = (destination / "results.md").read_bytes()
    with pytest.raises(FileExistsError, match="never overwritten"):
        publisher.publish(evidence, destination, write=True)
    assert (destination / "results.md").read_bytes() == before


def test_notebook_failure_does_not_leave_partial_publication(actual_format_fixture, tmp_path, monkeypatch):
    from nbclient import NotebookClient

    def fail(*args, **kwargs):
        raise RuntimeError("fixture notebook execution failed")

    monkeypatch.setattr(NotebookClient, "execute", fail)
    evidence = publisher.load_evidence(**actual_format_fixture)
    destination = tmp_path / "publication"
    with pytest.raises(RuntimeError, match="notebook execution failed"):
        publisher.publish(evidence, destination, write=True)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyclegan-publication-*"))
