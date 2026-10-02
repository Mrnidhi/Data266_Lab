"""The new search cannot promote a wider model within its practical tie band."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "sentiment_quality", Path(__file__).resolve().parents[1] / "scripts/run_sentiment_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def row(identity, family, f1, parameters):
    return {"candidate_id": identity, "model_name": family,
            "validation_macro_f1": f1, "parameter_count": parameters,
            "source_run": identity}


@pytest.fixture
def selection_case(tmp_path, monkeypatch):
    controls = [row("control_mlp", "maxpool_mlp", 0.95, 100),
                row("control_bilstm", "bilstm", 0.96, 200),
                row("control_cnn", "dilated_cnn", 0.95, 100)]
    candidates = {
        "mlp_regularized": row("mlp_regularized", "maxpool_mlp", 0.949, 100),
        "mlp_wide_regularized": row("mlp_wide_regularized", "maxpool_mlp", 0.951, 200),
        "cnn_regularized": row("cnn_regularized", "dilated_cnn", 0.952, 100),
        "cnn_wide_regularized": row("cnn_wide_regularized", "dilated_cnn", 0.9525, 200),
    }
    (tmp_path / "quality_plan.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(quality, "verify_plan", lambda output: {"controls": controls})
    monkeypatch.setattr(quality, "candidate_row", lambda output, plan, identity: dict(candidates[identity]))
    monkeypatch.setattr(quality, "inspect_source", lambda candidate: {})
    return tmp_path, candidates


def test_exact_tolerance_retains_smaller_control_and_all_candidates(selection_case):
    output, candidates = selection_case
    candidates["mlp_regularized"]["test_accuracy"] = 1.0
    result = quality.select(output)
    assert result["candidate_count"] == 7
    assert result["selected"]["maxpool_mlp"]["candidate_id"] == "control_mlp"
    assert result["selected"]["dilated_cnn"]["candidate_id"] == "cnn_regularized"
    assert result["test_metrics_used_for_selection"] is False
    assert result["test_evaluation_started"] is False


def test_wider_model_needs_more_than_tolerance(selection_case):
    output, candidates = selection_case
    candidates["mlp_wide_regularized"]["validation_macro_f1"] = 0.951001
    result = quality.select(output)
    assert result["selected"]["maxpool_mlp"]["candidate_id"] == "mlp_wide_regularized"


def test_missing_candidate_prevents_selection(selection_case):
    output, candidates = selection_case
    del candidates["cnn_wide_regularized"]
    with pytest.raises(KeyError):
        quality.select(output)
    assert not (output / "selection_manifest.json").exists()


def test_selection_cannot_be_replaced(selection_case):
    output, _ = selection_case
    quality.select(output)
    before = (output / "selection_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        quality.select(output)
    assert (output / "selection_manifest.json").read_bytes() == before
