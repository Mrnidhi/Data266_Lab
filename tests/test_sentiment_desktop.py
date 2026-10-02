"""Fixed desktop recipes must stay frozen before any test-set evaluation."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_sentiment_desktop", Path(__file__).resolve().parents[1] / "scripts/run_sentiment_desktop.py")
desktop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desktop)


@pytest.fixture
def planned(tmp_path):
    cfg = {"mode": "full", "seed": 2342, "epochs": 6, "embedding_dim": 128,
           "early_stopping_patience": 2, "data_dir": None}
    path = tmp_path / "task2_sentiment/srinidhi/config.json"
    desktop.write_json(path, {"full": cfg})
    for relative in desktop.RUNTIME_FILES:
        target = tmp_path / relative
        if target != path:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# Immutable synthetic runtime fixture\n", encoding="utf-8")
    output = tmp_path / "desktop"
    plan = desktop.freeze_plan(output, tmp_path)
    return tmp_path, output, plan


def test_plan_preserves_fixed_recipes_and_refuses_refreeze(planned):
    root, output, plan = planned
    assert desktop.verify_plan(output, root) == plan
    assert plan["base_config"]["embedding_dim"] == 128
    assert plan["historical_test_metrics_already_observed"] is True
    assert plan["recipes"]["bilstm"]["recipe"]["overrides"]["epochs"] == 12
    assert plan["recipes"]["bilstm"]["recipe"]["overrides"]["early_stopping_patience"] == 4
    assert all(row["recipe"]["overrides"]["num_workers"] == 0 for row in plan["recipes"].values())
    before = (output / "desktop_plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        desktop.freeze_plan(output, root)
    assert (output / "desktop_plan.json").read_bytes() == before


def test_changed_base_architecture_is_rejected(planned):
    root, output, _ = planned
    path = root / "task2_sentiment/srinidhi/config.json"
    config = desktop.finalizer.read_json(path)
    config["full"]["embedding_dim"] = 256
    desktop.write_json(path, config)
    with pytest.raises(ValueError, match="base configuration"):
        desktop.verify_plan(output, root)


def test_changed_training_source_is_rejected(planned):
    root, output, _ = planned
    path = root / "task2_sentiment/srinidhi/src/sentiment.py"
    path.write_text("# Changed synthetic source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training source"):
        desktop.verify_plan(output, root)


@pytest.mark.parametrize("update_embedded_hash", [False, True])
def test_recipe_tampering_cannot_change_prespecified_schedule(planned, update_embedded_hash):
    root, output, plan = planned
    path = root / plan["recipes"]["bilstm"]["path"]
    recipe = desktop.finalizer.read_json(path)
    recipe["overrides"]["epochs"] = 20
    desktop.write_json(path, recipe)
    if update_embedded_hash:
        plan["recipes"]["bilstm"].update(recipe=recipe, sha256=desktop.finalizer.digest(path))
        desktop.write_json(output / "desktop_plan.json", plan)
    with pytest.raises(ValueError, match="recipe"):
        desktop.verify_plan(output, root)


def test_plan_cannot_escape_repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    with pytest.raises(ValueError, match="inside the repository"):
        desktop.verify_plan(tmp_path / "outside", root)


@pytest.mark.parametrize("tamper", ["incomplete", "test_evaluated", "wrong_config", "wrong_candidate"])
def test_invalid_training_blocks_selection_before_source_verification(planned, monkeypatch, tamper):
    root, output, plan = planned
    name = "maxpool_mlp"
    run = output / "training" / name
    recipe = plan["recipes"][name]["recipe"]
    cfg = {**desktop.base_config(root), **recipe["overrides"], "validation_only": True}
    provenance = {"status": "completed", "candidate": recipe}
    metadata = {"test_evaluated": False}
    if tamper == "incomplete":
        provenance["status"] = "failed"
    elif tamper == "test_evaluated":
        metadata["test_evaluated"] = True
    elif tamper == "wrong_config":
        cfg["embedding_dim"] = 256
    else:
        provenance["candidate"] = {**recipe, "model": "bilstm"}
    desktop.write_json(run / "provenance.json", provenance)
    desktop.write_json(run / "config.json", cfg)
    desktop.write_json(run / name / "validation_selection.json", metadata)
    monkeypatch.setattr(desktop.finalizer, "verify_source", lambda *a, **kw: pytest.fail("Invalid source was verified"))
    with pytest.raises(ValueError, match="Unfinished or changed"):
        desktop.freeze_selection(output, root)
    assert not (output / "selection_manifest.json").exists()
