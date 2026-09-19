"""Selection is frozen from validation evidence, regardless of reference test scores."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import torch

from lab1 import sentiment as s
from lab1.common import project_root, task_config_path, write_json


SPEC = importlib.util.spec_from_file_location(
    "sentiment_selector", project_root() / "scripts/select_sentiment_candidates.py")
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def row(identity, score, count, path=None):
    return {"candidate_id": identity, "validation_macro_f1": score,
            "parameter_count": count, "source_run": path or identity}


@pytest.mark.parametrize("candidates,expected", [
    ([row("small", .9, 10), row("large", .901, 20)], "small"),
    ([row("small", .9, 10), row("large", .901001, 20)], "large"),
    ([row("original", .9, 10), row("control", .9008, 10)], "control"),
    # Apply tolerance to the global maximum: near ties must not chain transitively.
    ([row("small", .9, 10), row("medium", .9008, 20), row("large", .9016, 30)], "medium"),
    ([row("control", .9, 10, "runs/part-b-tuning/control"),
      row("reference", .9, 10, "runs/part-b-full")], "reference"),
])
def test_practical_tie_order(candidates, expected):
    chosen, _, _ = selector.choose_family(candidates)
    assert chosen["candidate_id"] == expected


def populate(root):
    """Create nine internally consistent tiny checkpoints; no training or inference."""
    torch.set_num_threads(1)
    base = json.loads(task_config_path("sentiment").read_text())["smoke"]
    base.update(mode="full", train_limit=None, validation_limit=None, test_limit=None)
    write_json(root / "task2_sentiment/srinidhi/config.json", {"full": base})
    recipe_dir = root / "task2_sentiment/srinidhi/experiments"
    recipe_dir.mkdir(parents=True)
    for recipe_name in selector.RECIPES:
        shutil.copy2(project_root() / "task2_sentiment/srinidhi/experiments" / f"{recipe_name}.json", recipe_dir)
    scores = {"reference:maxpool_mlp": .91, "maxpool_mlp_long_schedule_control": .9105, "mlp_wider": .9114,
              "reference:bilstm": .93, "bilstm_long_schedule_control": .932, "bilstm_wider": .9335,
              "reference:dilated_cnn": .944, "dilated_cnn_long_schedule_control": .945, "cnn_long_context": .9448}
    vocabulary = {"<pad>": 0, "<unk>": 1, "good": 2}
    reference_models = {}
    for identity, score in scores.items():
        reference = identity.startswith("reference:")
        name = identity.split(":")[1] if reference else selector.RECIPES[identity]
        run = root / "runs/part-b-full" if reference else root / "runs/part-b-tuning" / identity
        recipe = None if reference else json.loads((recipe_dir / f"{identity}.json").read_text())
        cfg = dict(base) if reference else {**base, **recipe["overrides"], "validation_only": True}
        model = s.build_model(name, len(vocabulary), cfg)
        history = [{"epoch": 1, "train_loss": .2, "validation_loss": .2,
                    "validation_macro_f1": score, "train_seconds": 1.5,
                    "learning_rate": cfg["learning_rates"][name]}]
        state = {"format_version": 2, "model_name": name, "config": cfg, "vocabulary": vocabulary,
                 "fingerprint": "synthetic-unit-fixture", "model": model.state_dict(),
                 "history": history, "epoch": 1, "best_f1": score, "global_step": 1,
                 "epoch_progress": None}
        checkpoint_dir = run / name / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        torch.save(state, checkpoint_dir / "best.pt")
        torch.save(state, checkpoint_dir / "last.pt")
        metadata = {"selected_epoch": 1, "selected_validation_macro_f1": score,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "train_seconds": 1.5, "completed_training_steps": 1, "peak_cuda_memory_bytes": None}
        write_json(run / "config.json", cfg)
        write_json(run / name / "history.json", history)
        if reference:
            metadata.update(accuracy=.999, macro={"f1": .999}, test_loss=.001)
            write_json(run / name / "metrics.json", metadata)
            reference_models[name] = metadata
        else:
            metadata["test_evaluated"] = False
            write_json(run / name / "validation_selection.json", metadata)
            write_json(run / "candidate.json", recipe)
            write_json(run / "provenance.json", {"status": "completed", "config": cfg,
                "environment": {"cpu_model": "Unit test CPU", "gpus": []}, "candidate": recipe, "result": metadata})
    write_json(root / "runs/part-b-full/run_summary.json", {"status": "completed", "config": base,
        "environment": {"cpu_model": "Unit test CPU", "gpus": []}, "summary": {"models": reference_models}})
    return base


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / "repo"
    populate(root)
    return root


def test_complete_ledger_and_finalizer_compatible_selection(source_root, tmp_path, monkeypatch):
    calls = []
    original = selector.finalizer.verify_source
    def verify(root, name, selected, base):
        calls.append(selected["candidate_id"])
        return original(root, name, selected, base)
    monkeypatch.setattr(selector.finalizer, "verify_source", verify)
    monkeypatch.setattr(s, "_predict", lambda *args, **kwargs: pytest.fail("Selection performed inference"))
    destination = tmp_path / "frozen.json"
    manifest = selector.select_candidates(destination, root=source_root)
    assert len(calls) == len(set(calls)) == manifest["candidate_count"] == 9
    assert manifest["required_completed_invocations"] == 7
    assert {name: selected["candidate_id"] for name, selected in manifest["selected"].items()} == {
        "maxpool_mlp": "maxpool_mlp_long_schedule_control", "bilstm": "bilstm_wider",
        "dilated_cnn": "dilated_cnn_long_schedule_control"}
    assert manifest["test_metrics_used_for_selection"] is False
    assert manifest["automatic_training_started"] is False
    assert [item["model_name"] for item in manifest["optional_seed_confirmations"]] == ["bilstm"]
    assert all(item["optional"] and not item["automatic_run_started"] for item in manifest["optional_seed_confirmations"])
    assert all(not item["statistical_significance_claimed"] for item in manifest["rationale"].values())
    assert all("decision" in candidate for candidate in manifest["candidates"])
    assert all("accuracy" not in candidate and "macro" not in candidate for candidate in manifest["candidates"])
    for name, selected in manifest["selected"].items():
        assert {"source_run", "checkpoint", "sha256", "config", "validation_macro_f1", "selected_epoch", "provenance"} <= selected.keys()
        state = torch.load(source_root / selected["checkpoint"], weights_only=False)
        assert selected["config"] == state["config"]
        assert selector.finalizer.digest(source_root / selected["checkpoint"]) == selected["sha256"]
    frozen_bytes = destination.read_bytes()
    with pytest.raises(FileExistsError):
        selector.select_candidates(destination, root=source_root)
    assert destination.read_bytes() == frozen_bytes


def test_reference_test_metrics_cannot_change_ranking(source_root, tmp_path):
    before = selector.select_candidates(tmp_path / "before.json", root=source_root)
    provenance_path = source_root / "runs/part-b-full/run_summary.json"
    provenance = selector.finalizer.read_json(provenance_path)
    for name in s.MODEL_NAMES:
        path = source_root / "runs/part-b-full" / name / "metrics.json"
        metrics = selector.finalizer.read_json(path)
        metrics.update(accuracy=0.0, macro={"f1": 0.0}, test_loss=9999.0)
        write_json(path, metrics)
        provenance["summary"]["models"][name] = metrics
    write_json(provenance_path, provenance)
    after = selector.select_candidates(tmp_path / "after.json", root=source_root)
    assert before["rationale"] == after["rationale"]
    assert [r["candidate_id"] for r in before["selected"].values()] == [r["candidate_id"] for r in after["selected"].values()]


@pytest.mark.parametrize("status", ["running", "failed", "missing"])
def test_all_seven_invocations_must_complete(source_root, tmp_path, monkeypatch, status):
    path = source_root / "runs/part-b-tuning/cnn_long_context/provenance.json"
    if status == "missing":
        path.unlink()
    else:
        provenance = selector.finalizer.read_json(path)
        provenance["status"] = status
        write_json(path, provenance)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Checkpoint opened before all invocations completed"))
    output = tmp_path / "not-created.json"
    with pytest.raises(ValueError, match="All seven invocations"):
        selector.select_candidates(output, root=source_root)
    assert not output.exists()


def test_mismatched_recipe_and_false_parameter_count_rejected(source_root, tmp_path):
    run = source_root / "runs/part-b-tuning/mlp_wider"
    recipe_path = run / "candidate.json"
    original_recipe = selector.finalizer.read_json(recipe_path)
    write_json(recipe_path, {**original_recipe, "name": "unlisted"})
    with pytest.raises(ValueError, match="explicit experiment recipe"):
        selector.select_candidates(tmp_path / "bad-recipe.json", root=source_root)
    write_json(recipe_path, original_recipe)
    metadata_path = run / "maxpool_mlp/validation_selection.json"
    metadata = selector.finalizer.read_json(metadata_path)
    metadata["parameter_count"] = 1
    write_json(metadata_path, metadata)
    provenance_path = run / "provenance.json"
    provenance = selector.finalizer.read_json(provenance_path)
    provenance["result"] = metadata
    write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match="Parameter count"):
        selector.select_candidates(tmp_path / "bad-size.json", root=source_root)
    assert not (tmp_path / "bad-recipe.json").exists()
    assert not (tmp_path / "bad-size.json").exists()
