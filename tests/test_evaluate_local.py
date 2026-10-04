import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluate_local", ROOT / "task3_gan/srinidhi/evaluate_local.py"
)
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


def test_official_evaluator_is_byte_identical():
    assert (
        hashlib.sha256(entrypoint.NOTEBOOK.read_bytes()).hexdigest() == entrypoint.EVALUATOR_SHA256
    )


def test_evaluation_never_overwrites_saved_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    evidence = results / "submission.csv"
    evidence.write_text("ID,FID,MiFID\n1,10,0.5\n")
    with pytest.raises(FileExistsError):
        entrypoint.validate_inputs(tmp_path / "data", tmp_path / "predictions", results)
    assert evidence.read_text() == "ID,FID,MiFID\n1,10,0.5\n"


def test_evaluation_rejects_incomplete_prediction_set(tmp_path):
    data = tmp_path / "data"
    (data / "monet_jpg").mkdir(parents=True)
    (data / "photo_jpg").mkdir()
    for index in range(300):
        (data / "monet_jpg" / f"{index}.jpg").touch()
    for index in range(7038):
        (data / "photo_jpg" / f"{index}.jpg").touch()
    with pytest.raises(ValueError, match="pred_A2B: expected 300 images, found 0"):
        entrypoint.validate_inputs(data, tmp_path / "predictions", tmp_path / "new-results")
    assert not (tmp_path / "new-results").exists()


def test_evaluation_rejects_modified_supplied_notebook(tmp_path, monkeypatch):
    altered = tmp_path / "altered.ipynb"
    altered.write_text("{}")
    monkeypatch.setattr(entrypoint, "NOTEBOOK", altered)
    with pytest.raises(ValueError, match="notebook has changed"):
        entrypoint.validate_inputs(tmp_path, tmp_path, tmp_path / "results")
