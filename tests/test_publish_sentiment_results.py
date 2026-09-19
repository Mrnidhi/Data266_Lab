"""Publication must retain source identity and any real review annotations."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "publish_sentiment_results", Path(__file__).resolve().parents[1] / "scripts/publish_sentiment_results.py"
)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def error_predictions():
    rows, records = [], []
    for group, label, probabilities, length in (
        ("fp", 0, [.99, .98, .97, .96, .95], 30),
        ("fn", 1, [.01, .02, .03, .04, .05], 40),
        ("near", 1, [.49, .48, .47, .46, .45], 50),
        ("long", 0, [.60, .61, .62, .63, .64], 250),
    ):
        for i, probability in enumerate(probabilities):
            example_id = f"{group}-{i}"
            rows.append({"example_id": example_id, "true_label": label,
                         "predicted_label": int(probability >= .5),
                         "positive_probability": probability, "original_token_length": length + i})
            records.append({"id": example_id, "text": f"Review {example_id}"})
    return pd.DataFrame(rows), records


def test_selects_five_distinct_errors_in_each_required_group():
    predictions, records = error_predictions()
    selected = publisher.select_errors(predictions, records)
    assert selected.example_id.nunique() == 20
    assert selected.groupby("review_group").size().to_dict() == {
        "confident_false_positive": 5, "confident_false_negative": 5,
        "near_threshold": 5, "long_review_slice": 5,
    }
    assert not selected.student_reviewed.any()
    assert selected.error_type.eq("").all() and selected.testable_fix.eq("").all()
    assert selected.text.notna().all()


def test_does_not_invent_missing_required_errors():
    predictions, records = error_predictions()
    with pytest.raises(ValueError, match="Fewer than five"):
        publisher.select_errors(predictions.iloc[:-1], records)


def test_republish_retains_real_annotations_and_archives_original(tmp_path):
    predictions, records = error_predictions()
    selected = publisher.select_errors(predictions, records)
    path = tmp_path / "required_20_errors_for_review.csv"
    publisher.write_error_review(path, selected, "checkpoint-a")
    prior = pd.read_csv(path, keep_default_na=False)
    prior.loc[0, ["error_type", "testable_fix", "student_reviewed"]] = ["mixed sentiment", "Retain more context", True]
    prior["reviewed_by"] = ""
    prior.loc[0, "reviewed_by"] = "student"
    prior.to_csv(path, index=False)
    annotated_bytes = path.read_bytes()

    publisher.write_error_review(path, selected, "checkpoint-a")
    saved = pd.read_csv(path, keep_default_na=False)
    assert saved.loc[0, "error_type"] == "mixed sentiment"
    assert saved.loc[0, "testable_fix"] == "Retain more context"
    assert saved.loc[0, "student_reviewed"]
    assert saved.loc[0, "reviewed_by"] == "student"
    assert not saved.loc[1:, "student_reviewed"].any()
    assert next((tmp_path / "review_history").glob("*.csv")).read_bytes() == annotated_bytes


def test_changed_checkpoint_does_not_inherit_human_review(tmp_path):
    predictions, records = error_predictions()
    selected = publisher.select_errors(predictions, records)
    path = tmp_path / "required_20_errors_for_review.csv"
    reviewed = selected.copy()
    reviewed.loc[0, "error_type"] = "negation"
    reviewed.loc[0, "student_reviewed"] = True
    publisher.write_error_review(path, reviewed, "old-checkpoint")
    publisher.write_error_review(path, selected, "new-checkpoint")
    refreshed = pd.read_csv(path, keep_default_na=False)
    assert not refreshed.student_reviewed.any()
    assert refreshed.error_type.eq("").all()
    assert len(list((tmp_path / "review_history").glob("*.csv"))) == 1


def test_uses_selected_model_config_and_training_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    hardware = tmp_path / "verification/part_b_cpu_hardware.txt"
    hardware.parent.mkdir()
    hardware.write_text("Architecture: x86_64\nModel name: AMD EPYC example\n")
    selection = {"selection_rule": "highest validation macro-F1", "selected": {
        name: {"source_run": f"runs/{name}", "checkpoint": f"{name}/checkpoints/best.pt",
               "config": {"embedding_dim": 128 + i}, "provenance": {
                   "environment": {"gpus": [{"name": f"Training GPU {i}"}]}}}
        for i, name in enumerate(publisher.NAMES)
    }}
    sources = publisher.model_sources(tmp_path / "runs/final", {"embedding_dim": 999},
                                      {"environment": {"gpus": [{"name": "Evaluation GPU"}]}}, selection)
    assert sources["bilstm"]["config"]["embedding_dim"] == 129
    assert sources["bilstm"]["hardware"]["gpu"] == "Training GPU 1"
    assert sources["bilstm"]["hardware"]["cpu"] == "AMD EPYC example"
    assert sources["bilstm"]["checkpoint"] == "runs/bilstm/bilstm/checkpoints/best.pt"
    assert sources["bilstm"]["source_run"] == "runs/bilstm"
    assert "hardware" not in selection["selected"]["bilstm"]


def test_reproduction_uses_preserved_candidate_and_explains_finalizer(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    run = tmp_path / "runs/selected"
    run.mkdir(parents=True)
    source = tmp_path / "runs/candidate"
    source.mkdir()
    candidate = {"model": "bilstm", "overrides": {"dropout": .2}}
    (source / "candidate.json").write_text(json.dumps(candidate))
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    text = publisher.reproduction_text(run, outputs, {
        "bilstm": {"source_run": "runs/candidate", "config": {"dropout": .2}}}, {"selected": {}})
    assert "scripts/tune_sentiment.py --candidate outputs/reproduction/bilstm_candidate.json" in text
    assert "scripts/finalize_sentiment_selection.py --selection runs/selected/selection_manifest.json" in text
    assert "does not select the newly trained runs" in text
    assert json.loads((outputs / "reproduction/bilstm_candidate.json").read_text()) == candidate


def test_copies_distinct_logs_with_source_mapping_and_unchanged_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    originals = {
        "runs/first/candidate": b"first training\r\nraw bytes\n",
        "runs/second/candidate": b"second training\n",
        "runs/evaluation": b"final evaluation\n",
    }
    for relative, content in originals.items():
        folder = tmp_path / relative
        folder.mkdir(parents=True)
        (folder / "RUN_LOG.txt").write_bytes(content)
    sources = {
        "maxpool_mlp": {"source_run": "runs/first/candidate"},
        "bilstm": {"source_run": "runs/first/candidate"},
        "dilated_cnn": {"source_run": "runs/second/candidate"},
    }
    manifest = publisher.copy_raw_logs(tmp_path / "runs/evaluation", outputs, sources)
    assert len(manifest["logs"]) == 3
    assert len({row["published_path"] for row in manifest["logs"]}) == 3
    assert {row["role"] for row in manifest["logs"]} == {"training", "evaluation"}
    for row in manifest["logs"]:
        expected = originals[row["source_run"]]
        assert (tmp_path / row["source_path"]).read_bytes() == expected
        assert (outputs / row["published_path"]).read_bytes() == expected
        assert row["sha256"] == hashlib.sha256(expected).hexdigest()
        assert row["bytes"] == len(expected)
    shared = next(row for row in manifest["logs"] if row["source_run"] == "runs/first/candidate")
    assert shared["models"] == ["bilstm", "maxpool_mlp"]
    assert json.loads((outputs / "raw_logs/manifest.json").read_text()) == manifest


def test_notebook_presentation_executes_without_original_runs(tmp_path, monkeypatch):
    import IPython.display

    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    member = tmp_path / "task2_sentiment/srinidhi"
    outputs = member / "outputs/full"
    outputs.mkdir(parents=True)
    original = tmp_path / "runs/original"
    original.mkdir(parents=True)
    (original / "RUN_LOG.txt").write_text("Original training and evaluation log\n")
    sources = {name: {"source_run": "runs/original", "hardware": {"cpu": "fixture CPU"},
                      "config": {"embedding_dim": 128}} for name in publisher.NAMES}
    publisher.copy_raw_logs(original, outputs, sources)
    publisher.write_json(outputs / "training_sources.json", sources)
    publisher.write_json(outputs / "summary.json", {"mode": "full", "models": {}})
    publisher.write_json(outputs / "run_summary.json", {"status": "completed", "evaluation_environment": {"device": "cpu"}})
    publisher.write_json(outputs / "data_audit.json", {"split_counts": {"test": 38000}})
    selection = {"selection_rule": "validation macro-F1", "selected": {
        name: {"source_run": "runs/original", "selected_epoch": 2, "validation_macro_f1": .9,
               "sha256": str(index) * 64, "provenance": {"large_environment": list(range(1000))}}
        for index, name in enumerate(publisher.NAMES)
    }}
    publisher.write_json(outputs / "selection_manifest.json", selection)
    preserved_selection = (outputs / "selection_manifest.json").read_bytes()
    publisher.write_json(outputs / "data_distributions.json", {
        "length_definition": "Processed tokens before truncation", "splits": {"train": {"count": 504000}}})
    (outputs / "data_distributions.png").write_bytes(b"EDA image fixture")
    table = pd.DataFrame({"model": list(publisher.NAMES), "accuracy": [.8, .9, .9]})
    table.to_csv(outputs / "model_comparison.csv", index=False)
    table.to_csv(member / "metrics_report.csv", index=False)
    for name in publisher.NAMES:
        folder = outputs / name
        folder.mkdir()
        publisher.write_json(folder / "training_config.json", {"embedding_dim": 128})
        (folder / "learning_curves.png").write_bytes(b"curve image fixture")
        pd.DataFrame({"example_id": ["review"], "student_reviewed": [False]}).to_csv(
            folder / "required_20_errors_for_review.csv", index=False)

    # A fresh clone has published artifacts but no ignored original runs tree.
    (tmp_path / "runs").rename(tmp_path / "unavailable_original_runs")
    displayed, images = [], []
    monkeypatch.setattr(IPython.display, "display", displayed.append)

    def checked_image(filename, width):
        path = Path(filename)
        assert path.is_relative_to(outputs)
        assert path.read_bytes()
        images.append(path)
        return (path.name, width)

    monkeypatch.setattr(IPython.display, "Image", checked_image)
    namespace = {}
    notebook = publisher.build_notebook("Preserved reproduction commands")
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        compile(cell.source, "published_notebook", "exec")
        if "from lab1.sentiment import" not in cell.source:
            exec(cell.source, namespace)
    assert not (tmp_path / "runs").exists()
    assert len(images) == 4
    compact = [value for value in displayed if isinstance(value, pd.DataFrame) and "checkpoint_sha256" in value]
    assert len(compact) == 1 and len(compact[0]) == 3
    assert list(compact[0].columns) == ["model", "source_run", "selected_epoch", "validation_macro_f1", "checkpoint_sha256"]
    assert not any(isinstance(value, dict) and "selected" in value for value in displayed)
    assert (outputs / "selection_manifest.json").read_bytes() == preserved_selection
