"""Portable Part 2 publication preserves history and rejects altered evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import nbformat
import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import finalize_part2 as pipeline
import publish_sentiment_results as publisher
import verify_part2_notebook as notebook_runner
from lab1 import sentiment as s


def test_windows_native_and_mixed_metadata_paths_preserve_other_text(tmp_path):
    root = tmp_path / "repository"
    for separator in ("/", "\\"):
        value = {"cache": str(root) + separator + "data\\encoded", "unrelated": "Text uses \\n literally"}
        portable = publisher.normalize_repository_paths(value, root)
        assert portable == {"cache": "data/encoded", "unrelated": value["unrelated"]}
        assert value["cache"].startswith(str(root))


def test_unreviewed_previous_error_sets_are_preserved(tmp_path):
    path = tmp_path / "required_20_errors_for_review.csv"
    rows = pd.DataFrame({"example_id": ["review1"], "true_label": [0], "predicted_label": [1],
                         "positive_probability": [.9], "error_type": [""], "testable_fix": [""],
                         "student_reviewed": [False]})
    publisher.write_error_review(path, rows, "old")
    before = path.read_bytes()
    publisher.write_error_review(path, rows, "new")
    archived = list((tmp_path / "review_history").glob("*.csv"))
    assert len(archived) == 1 and archived[0].read_bytes() == before
    assert not pd.read_csv(path).student_reviewed.any()


def test_previous_publication_snapshot_preserves_all_bytes_and_is_idempotent(tmp_path):
    member = tmp_path / "task2_sentiment/srinidhi"
    paths = {"outputs/full/summary.json": b'{"old": true}\n',
             "outputs/full/publication_metadata.json": b'{"source_run": "runs/old"}\n',
             "outputs/full/bilstm/required_20_errors_for_review.csv": b"student_reviewed,ai_draft\nFalse,negation\n",
             "metrics_report.csv": b"metric,value\naccuracy,.9\n", "results.md": b"Old observed results\n",
             "src/sentiment.ipynb": b"Original notebook bytes\n"}
    for relative, content in paths.items():
        path = member / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    archive = publisher.archive_previous_publication(member, "runs/new")
    for relative, content in paths.items():
        assert (member / relative).read_bytes() == content
        assert (archive / relative).read_bytes() == content
    assert publisher.archive_previous_publication(member, "runs/new") == archive
    publisher.write_json(member / "outputs/full/publication_metadata.json", {"source_run": "runs/new"})
    assert publisher.archive_previous_publication(member, "runs/new") is None


def test_console_host_paths_are_preserved_locally_and_epoch_bytes_are_portable(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    training = tmp_path / "runs/training"
    evaluation = tmp_path / "runs/selected"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    training.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    original = b"warning from C:\\Users\\student\\environment\\library.py\ntraining complete\n"
    (training / "RUN_LOG.txt").write_bytes(original)
    for name in publisher.NAMES:
        folder = training / name
        folder.mkdir()
        (folder / "history.json").write_bytes(b'[{"epoch":1,"train_loss":0.2}]\n')
    (evaluation / "RUN_LOG.txt").write_bytes(b"evaluation complete\n")
    sources = {name: {"source_run": "runs/training"} for name in publisher.NAMES}
    result = publisher.copy_raw_logs(evaluation, outputs, sources)
    assert (training / "RUN_LOG.txt").read_bytes() == original
    assert len(result["omitted_console_files"]) == 1
    assert len(result["logs"]) == 4
    assert all(b"Users" not in (outputs / row["published_path"]).read_bytes() for row in result["logs"])


def test_metric_recomputation_rejects_accuracy_and_bootstrap_changes():
    y = np.asarray([0, 0, 1, 1] * 5)
    probability = np.asarray([.1, .8, .7, .2] * 5)
    predictions = pd.DataFrame({"true_label": y, "positive_probability": probability,
                                "predicted_label": (probability >= .5).astype(int)})
    metrics, _ = s.compute_metrics(y, probability, bootstrap_samples=20, seed=2342)
    pipeline.verify_prediction_metrics(predictions, metrics, seed=2342, bootstrap_samples=20)
    metrics["accuracy"] += .01
    with pytest.raises(ValueError, match="accuracy value differs"):
        pipeline.verify_prediction_metrics(predictions, metrics, seed=2342, bootstrap_samples=20)
    metrics["accuracy"] -= .01
    metrics["bootstrap_95_percentile"]["accuracy"]["low"] += .01
    with pytest.raises(ValueError, match="bootstrap_95_percentile.accuracy.low"):
        pipeline.verify_prediction_metrics(predictions, metrics, seed=2342, bootstrap_samples=20)


def test_notebook_validation_requires_outputs_and_no_errors():
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('ok')", execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text="ok\n")])],
        metadata={"lab1": {"mode": "completed_full_run_evidence"}})
    assert notebook_runner.validate_notebook(notebook)["executed_code_cells"] == 1
    notebook.cells[0].outputs = []
    with pytest.raises(ValueError, match="visible output"):
        notebook_runner.validate_notebook(notebook)


def package_fixture(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    run_relative = "reproducibility/raw_logs/srinidhi/desktop-test/part2-full/selected"
    run = root / run_relative
    data_dir = root / "task2_sentiment/srinidhi/data_processed/full"
    features = root / "task2_sentiment/srinidhi/data_processed/full_encoded"
    required = ["src/lab1/__init__.py", "src/lab1/common.py", "src/lab1/run.py",
                "task2_sentiment/srinidhi/src/sentiment.py", "task2_sentiment/srinidhi/config.json",
                "task2_sentiment/srinidhi/metrics_report.csv", "task2_sentiment/srinidhi/checkpoints/manifest.json", "scripts/finalize_part2.py",
                "scripts/publish_sentiment_results.py", "scripts/verify_part2_notebook.py", "scripts/tune_sentiment.py"]
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('ok')", execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text="ok\n")])],
        metadata={"lab1": {"mode": "completed_full_run_evidence"}})
    nbformat.write(notebook, root / "task2_sentiment/srinidhi/src/sentiment.ipynb")
    for folder, suffix in ((data_dir, "jsonl"), (features, "npz")):
        folder.mkdir(parents=True)
        for split in ("train", "validation", "test"):
            (folder / f"{split}.{suffix}").write_bytes(b"fixture split\n")
        publisher.write_json(folder / "manifest.json", {})
    selected = {}
    for name in s.MODEL_NAMES:
        source_relative = f"reproducibility/raw_logs/srinidhi/desktop-test/part2-full/training/{name}"
        checkpoint = f"{source_relative}/{name}/checkpoints/best.pt"
        for location in (root / source_relative / name / "checkpoints", root / "task2_sentiment/srinidhi/checkpoints" / name):
            location.mkdir(parents=True)
            for filename in ("best.pt", "last.pt"):
                (location / filename).write_bytes(b"fixture trained state\n")
        selected[name] = {"source_run": source_relative, "checkpoint": checkpoint,
                          "sha256": pipeline.digest(root / checkpoint)}
        review = root / "task2_sentiment/srinidhi/outputs/full" / name / "required_20_errors_for_review.csv"
        review.parent.mkdir(parents=True)
        review.write_bytes(b"fixture review\n")
    run.mkdir(parents=True)
    publisher.write_json(run / "selection_manifest.json", {"selected": selected})
    for filename in ("config.json", "resolved_config.json", "run_summary.json"):
        publisher.write_json(run / filename, {"data_dir": str(data_dir), "accuracy": .9})
    for filename in ("summary.json", "data_audit.json"):
        publisher.write_json(run / filename, {})
    return root, {"run": run_relative, "data_dir": data_dir, "feature_cache": features,
                  "selection": {"selected": selected}}


def test_archive_contains_portable_runtime_six_weights_and_exact_data(tmp_path):
    root, evidence = package_fixture(tmp_path)
    before = pipeline.inventory(root, [root / evidence["run"]])
    archive = root / "dist/Part2.zip"
    result = pipeline.create_archive(root, evidence, archive)
    assert result["crc_and_content_hashes_verified"]
    assert pipeline.inventory(root, [root / evidence["run"]]) == before
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert all("\\" not in name for name in names)
        assert b"torchvision" not in handle.read("Part 2/pyproject.toml")
        config = json.loads(handle.read(f"Part 2/{evidence['run']}/config.json"))
        assert config["data_dir"] == "task2_sentiment/srinidhi/data_processed/full"
        manifest = json.loads(handle.read("Part 2/PACKAGE_MANIFEST.json"))
        assert len(manifest["derived_metadata_copies"]) == 3
        entries = {name: handle.read(name) for name in names}
    entries["Part 2/task2_sentiment/srinidhi/checkpoints/bilstm/best.pt"] = b"altered weight"
    damaged = root / "dist/damaged.zip"
    with zipfile.ZipFile(damaged, "w") as handle:
        for name, value in entries.items():
            handle.writestr(name, value)
    with pytest.raises(ValueError, match="SHA256/size mismatch"):
        pipeline.verify_archive(damaged)


def test_archive_refuses_mutating_evidence_and_path_traversal(tmp_path):
    root, evidence = package_fixture(tmp_path)
    with pytest.raises(ValueError, match="immutable"):
        pipeline.create_archive(root, evidence, root / evidence["run"] / "unsafe.zip")
    archive = tmp_path / "traversal.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Part 2/../outside", "unsafe")
    with pytest.raises(ValueError, match="Unsafe or noncanonical"):
        pipeline.verify_archive(archive)


def test_portability_commands_use_extracted_imports_without_mutating_originals(tmp_path, monkeypatch):
    root, evidence = package_fixture(tmp_path)
    tuner = root / "scripts/tune_sentiment.py"
    tuner.write_bytes(b"# Fixture tuner\n")
    archive = root / "dist/Part2.zip"
    pipeline.create_archive(root, evidence, archive)
    before = (root / "task2_sentiment/srinidhi/src/sentiment.ipynb").read_bytes()
    calls = []
    def child(command, *, cwd, env, **kwargs):
        calls.append(command)
        assert Path(env["PYTHONPATH"]) == cwd / "src"
        assert cwd != root and cwd.name == "Part 2"
        if "scripts/verify_part2_notebook.py" in command:
            publisher.write_json(cwd / "verification/part2_notebook.json", {"executed_code_cells": 1, "errors": 0})
            publisher.write_json(cwd / "verification/part2_checkpoint_inference.json", {
                "models": {name: {"devices": [{"device": "cpu", "finite_logits": True}]} for name in s.MODEL_NAMES}})
        else:
            publisher.write_json(cwd / "runs/part2-package-smoke/run_summary.json", {
                "status": "completed", "mode": "smoke", "summary": {"models": {name: {} for name in s.MODEL_NAMES}}})
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(pipeline.subprocess, "run", child)
    receipt = pipeline.verify_portability(root, archive)
    assert len(calls) == 2 and receipt["imports_isolated_with_extracted_src"]
    assert (root / "task2_sentiment/srinidhi/src/sentiment.ipynb").read_bytes() == before


def archive_only_fixture(tmp_path):
    root, evidence = package_fixture(tmp_path)
    run = root / evidence["run"]
    summary = {"models": {name: {} for name in s.MODEL_NAMES},
               "selection_manifest_sha256": pipeline.digest(run / "selection_manifest.json")}
    publisher.write_json(run / "summary.json", summary)
    publisher.write_json(run / "config.json", {"encoded_cache": str(evidence["feature_cache"])})
    publisher.write_json(run / "run_summary.json", {"status": "completed", "summary": summary,
                         "selection_manifest_sha256": summary["selection_manifest_sha256"]})
    directories = [run, *(root / row["source_run"] for row in evidence["selection"]["selected"].values())]
    receipt = {"source_run": evidence["run"], "raw_evidence_unchanged": True, "recomputed_test_metrics": True,
               "data_dir": evidence["data_dir"].relative_to(root).as_posix(),
               "feature_cache": evidence["feature_cache"].relative_to(root).as_posix(),
               "raw_files": pipeline.inventory(root, directories), "frozen_inputs": pipeline.frozen_inputs(root, evidence),
               "published_technical_files": pipeline.technical_publication_files(root)}
    publisher.write_json(root / "verification/part2_finalization.json", receipt)
    return root, evidence


def test_archive_only_collects_added_ai_drafts_without_republishing_or_execution(tmp_path, monkeypatch):
    root, evidence = archive_only_fixture(tmp_path)
    draft = root / "task2_sentiment/srinidhi/outputs/full/bilstm/ai_error_review_draft.csv"
    draft.write_bytes(b"example_id,ai_only_draft\nreview1,True\n")
    notebook_path = root / "task2_sentiment/srinidhi/src/sentiment.ipynb"
    notebook_before = notebook_path.read_bytes()
    raw_before = pipeline.inventory(root, [root / evidence["run"]])
    def forbidden(*args, **kwargs):
        pytest.fail("Archive-only must reuse prior metric/execution proof")
    monkeypatch.setattr(pipeline, "verify_evaluation", forbidden)
    monkeypatch.setattr(publisher, "publish", forbidden)
    monkeypatch.setattr(notebook_runner, "execute_notebook", forbidden)
    monkeypatch.setattr(pipeline, "verify_publication", lambda root, evidence: {"weights_verified": True})
    archive = root / "dist/Part2.zip"
    pipeline.archive_only(root, root / evidence["run"], evidence["data_dir"], destination=archive)
    with zipfile.ZipFile(archive) as handle:
        assert handle.read("Part 2/" + draft.relative_to(root).as_posix()) == draft.read_bytes()
    assert notebook_path.read_bytes() == notebook_before
    assert pipeline.inventory(root, [root / evidence["run"]]) == raw_before
    receipt = pipeline.read_json(root / "verification/part2_archive_only.json")
    assert receipt["bootstrap_recomputed"] is False and receipt["publication_recreated"] is False


@pytest.mark.parametrize("changed", ["raw_checkpoint", "frozen_data", "runtime", "figure"])
def test_archive_only_rejects_changed_completed_evidence_before_any_output(tmp_path, monkeypatch, changed):
    root, evidence = archive_only_fixture(tmp_path)
    paths = {"raw_checkpoint": root / evidence["selection"]["selected"]["bilstm"]["checkpoint"],
             "frozen_data": evidence["data_dir"] / "test.jsonl", "runtime": root / "scripts/tune_sentiment.py",
             "figure": root / "task2_sentiment/srinidhi/metrics_report.csv"}
    paths[changed].write_bytes(b"changed after successful finalization\n")
    def forbidden(*args, **kwargs):
        pytest.fail("Changed immutable evidence must fail before publication or packaging")
    monkeypatch.setattr(pipeline, "verify_publication", forbidden)
    monkeypatch.setattr(pipeline, "create_archive", forbidden)
    with pytest.raises(ValueError, match="bytes changed"):
        pipeline.archive_only(root, root / evidence["run"], evidence["data_dir"])
    assert not (root / "verification/part2_archive_only.json").exists()
