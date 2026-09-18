import csv
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_local", ROOT / "task3_gan/srinidhi/evaluate_local.py")
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


def test_report_preserves_missing_values_and_protects_existing_results(tmp_path):
    result = {"checkpoint_sha256": "test-digest", "directions": {"photo_to_monet": {"metrics": {
        "fid": {"status": "unavailable", "value": None, "reason": "synthetic fixture"},
        "cycle_l1": {"status": "ok", "value": 0.25, "count": 2}}}}}
    report = tmp_path / "full_metrics_report.csv"
    entrypoint.write_metrics_report(result, report)
    rows = list(csv.DictReader(report.open()))
    assert rows[0]["value"] == "" and rows[0]["status"] == "unavailable"
    assert "synthetic fixture" in rows[0]["details"]
    assert rows[1]["value"] == "0.25" and rows[1]["checkpoint_sha256"] == "test-digest"
    original = report.read_bytes()
    with pytest.raises(FileExistsError):
        entrypoint.write_metrics_report(result, report)
    assert report.read_bytes() == original
