"""Frozen-selection checks must fail before test inference or output publication."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lab1 import sentiment as s
from lab1.common import project_root, task_config_path, write_json


SPEC = importlib.util.spec_from_file_location(
    "sentiment_finalizer", project_root() / "scripts/finalize_sentiment_selection.py")
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("frozen-selection")
    base = json.loads(task_config_path("sentiment").read_text())["smoke"]
    base.update(cpu_threads=1, bootstrap_samples=5)
    torch.set_num_threads(1)
    cache = root / "encoded"
    s.prepare_features(base, cache)
    base["encoded_cache"] = str(cache)
    records = s._load_records(base)
    vocabulary, datasets = s._load_features(base, records)
    selected = {}
    for index, name in enumerate(s.MODEL_NAMES):
        run = root / "runs" / name
        cfg = dict(base, validation_only=True, model_seed=44 + index, mlp_hidden=10 + index)
        fingerprint = s._fingerprint(cfg, vocabulary, records)
        metrics, _, _ = s._train_one(name, cfg, {key: datasets[key] for key in ("train", "validation")},
                                     vocabulary, run, "cpu", fingerprint, None)
        provenance = {"status": "completed", "config": cfg, "result": metrics,
                      "environment": {"cpu_model": "Synthetic test CPU", "gpus": []}}
        write_json(run / "config.json", cfg)
        write_json(run / "provenance.json", provenance)
        checkpoint = run / name / "checkpoints/best.pt"
        selected[name] = {"source_run": run.relative_to(root).as_posix(),
                          "checkpoint": checkpoint.relative_to(run).as_posix(),
                          "sha256": finalizer.digest(checkpoint), "config": cfg,
                          "validation_macro_f1": metrics["selected_validation_macro_f1"],
                          "selected_epoch": metrics["selected_epoch"], "provenance": provenance}
    manifest = {"selection_rule": "Maximum validation macro-F1, then smaller model",
                "candidates": list(selected), "selected": selected}
    selection = root / "selection.json"
    write_json(selection, manifest)
    return root, base, selection, datasets, vocabulary


def test_selected_checkpoint_evaluation_matches_original_exports(tmp_path, frozen, monkeypatch):
    root, base, selection, datasets, vocabulary = frozen
    entry = json.loads(selection.read_text())["selected"]["bilstm"]
    checkpoint = root / entry["source_run"] / entry["checkpoint"]
    state = torch.load(checkpoint, weights_only=False)
    model = s.build_model("bilstm", len(vocabulary), state["config"])
    model.load_state_dict(state["model"])
    expected_y, expected_p, expected_loss = s._predict(
        model, s._loader(datasets["test"], base["batch_size"]), "cpu")
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: pytest.fail("Evaluation trained a model"))
    metrics, y, probability = s.evaluate_checkpoint(checkpoint, datasets["test"], tmp_path)
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(probability, expected_p)
    assert metrics["test_loss"] == expected_loss
    assert metrics["selected_epoch"] == entry["selected_epoch"]
    assert metrics["selected_validation_macro_f1"] == entry["validation_macro_f1"]
    for filename in ("metrics.json", "curves.json", "slices.json", "test_predictions.csv",
                     "error_candidates_for_manual_review.csv", "learning_curves.png", "confusion_matrix.png"):
        assert (tmp_path / filename).is_file()


def test_finalizer_preserves_sources_and_pairs_predictions(tmp_path, frozen, monkeypatch):
    root, base, selection, datasets, vocabulary = frozen
    monkeypatch.setattr(finalizer, "environment", lambda: {"cpu_model": "Evaluation CPU", "gpus": []})
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: pytest.fail("Finalizer trained"))
    output = tmp_path / "derived"
    summary = finalizer.finalize(selection, output, root=root, base_config=base)
    assert summary["synthetic"] and set(summary["models"]) == set(s.MODEL_NAMES)
    assert (output / "selection_manifest.json").read_bytes() == selection.read_bytes()
    provenance = finalizer.read_json(output / "run_summary.json")
    assert provenance["status"] == "completed" and provenance["assembled_results"] is True
    assert provenance["environment"]["cpu"]["details"] == "Synthetic test CPU"
    assert provenance["evaluation_environment"]["cpu_model"] == "Evaluation CPU"
    config = finalizer.read_json(output / "config.json")
    assert config["model_configs"]["bilstm"]["model_seed"] == 45
    assert len({cfg["mlp_hidden"] for cfg in config["model_configs"].values()}) == 3
    for name in s.MODEL_NAMES:
        source = finalizer.read_json(output / name / "source_selection.json")
        assert finalizer.digest(output / name / "checkpoints/best.pt") == source["sha256"]
        assert (output / name / "checkpoints/last.pt").exists()
        assert (root / source["resolved_checkpoint"]).is_file()
    for comparison in summary["paired_mcnemar"].values():
        assert sum(map(sum, comparison["table"])) == base["test_limit"]
    with np.load(output / "split_ids.npz") as splits:
        assert splits["test"].tolist() == [r["id"] for r in datasets["test"].records]
    with pytest.raises(FileExistsError):
        finalizer.finalize(selection, output, root=root, base_config=base)


@pytest.mark.parametrize("field,value,match", [
    ("sha256", "0" * 64, "SHA256"),
    ("selected_epoch", 9, "score/epoch"),
    ("validation_macro_f1", 0.123, "score/epoch"),
    ("provenance", {"status": "completed"}, "provenance"),
])
def test_any_invalid_selected_source_blocks_all_test_inference(tmp_path, frozen, monkeypatch, field, value, match):
    root, base, selection, _, _ = frozen
    manifest = finalizer.read_json(selection)
    # Tamper with the last model so preceding selected models must not be evaluated.
    manifest["selected"]["dilated_cnn"][field] = value
    changed = tmp_path / "changed.json"
    write_json(changed, manifest)
    monkeypatch.setattr(s, "_predict", lambda *a, **kw: pytest.fail("Test inference ran before validation completed"))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=match):
        finalizer.finalize(changed, output, root=root, base_config=base)
    assert not output.exists()


def test_changed_preprocessing_is_rejected_before_inference(tmp_path, frozen, monkeypatch):
    root, base, selection, _, _ = frozen
    monkeypatch.setattr(s, "_predict", lambda *a, **kw: pytest.fail("Unexpected inference"))
    with pytest.raises(ValueError, match="preprocessing/split"):
        finalizer.finalize(selection, tmp_path / "bad", root=root, base_config=dict(base, max_length=16))


def test_original_suite_source_metadata_is_supported(tmp_path, frozen):
    _, base, _, _, _ = frozen
    run = tmp_path / "baseline"
    result = s.run(base, run, "cpu")
    provenance = {"status": "completed", "config": base, "summary": result,
                  "environment": {"cpu_model": "Synthetic test CPU", "gpus": []}}
    write_json(run / "run_summary.json", provenance)
    for name in s.MODEL_NAMES:
        checkpoint = run / name / "checkpoints/best.pt"
        state = torch.load(checkpoint, weights_only=False)
        selected = {"source_run": "baseline", "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
                    "sha256": finalizer.digest(checkpoint), "config": state["config"],
                    "validation_macro_f1": state["best_f1"], "selected_epoch": state["epoch"],
                    "provenance": provenance}
        verified = finalizer.verify_source(tmp_path, name, selected, base)
        assert verified["metadata"] == result["models"][name]


def test_model_seed_changes_initialization_without_changing_split(tmp_path, frozen, monkeypatch):
    _, base, _, datasets, vocabulary = frozen
    initial = []
    original = s.build_model
    def capture(*args):
        model = original(*args)
        initial.append(model.embedding.weight.detach().clone())
        return model
    monkeypatch.setattr(s, "build_model", capture)
    for seed in (21, 22):
        cfg = dict(base, validation_only=True, model_seed=seed)
        assert s._load_records(cfg) == s._load_records(base)
        s._train_one("maxpool_mlp", cfg, {key: datasets[key] for key in ("train", "validation")},
                     vocabulary, tmp_path / str(seed), "cpu", "test-seed", None)
    assert not torch.equal(initial[0], initial[1])
