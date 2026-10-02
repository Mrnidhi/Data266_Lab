"""Portable receipts remain explicitly derived and cannot rewrite raw evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "quality_evidence", Path(__file__).resolve().parents[1] / "scripts/export_sentiment_quality_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


@pytest.fixture
def original(tmp_path):
    run, output = tmp_path / "run", tmp_path / "published"
    (run / "invocations").mkdir(parents=True)
    plan = run / "quality_plan.json"
    plan.write_text(json.dumps({"recipes": {"candidate": {}}}), encoding="utf-8")
    (run / "selection_manifest.json").write_text(json.dumps({
        "test_metrics_used_for_selection": False, "quality_plan_sha256": evidence.digest(plan)}), encoding="utf-8")
    receipt = run / "invocations/candidate.json"
    receipt.write_text(json.dumps({"candidate_id": "candidate", "ended_utc": "2026-10-02T00:00:00Z", "returncode": 0,
                                   "command": ["C:\\private\\python.exe", "scripts/tune_sentiment.py", "--candidate", "recipe.json"]}), encoding="utf-8")
    return tmp_path, run, output, receipt


def test_portable_copy_preserves_raw_bytes_and_records_hash(original):
    root, run, output, receipt = original
    before = receipt.read_bytes()
    manifest = evidence.export(run, output, root)
    published = root / manifest["receipts"][0]["published_receipt"]
    derived = evidence.read_json(published)
    assert receipt.read_bytes() == before
    assert derived["command"][0] == "python"
    assert derived["derived_evidence"]["source_sha256"] == evidence.digest(receipt)
    assert "private" not in published.read_text(encoding="utf-8")
    assert evidence.export(run, output, root) == manifest


def test_unfinished_invocation_blocks_export(original):
    root, run, output, receipt = original
    value = evidence.read_json(receipt)
    value.pop("returncode")
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="completed successfully"):
        evidence.export(run, output, root)
    assert not output.exists()


def test_undeclared_external_path_blocks_export(original):
    root, run, output, receipt = original
    value = evidence.read_json(receipt)
    value["extra_path"] = "C:\\private\\more.txt"
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="absolute path remains"):
        evidence.export(run, output, root)
    assert not output.exists()
