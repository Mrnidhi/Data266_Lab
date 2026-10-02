"""Guard the data/context contract that makes GPT comparison meaningful."""
import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/compare_gpt_candidates.py"
SPEC = importlib.util.spec_from_file_location("compare_gpt_candidates", PATH)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


@pytest.fixture
def bundles():
    first = {"summary": {"manifest_sha256": "frozen-split"},
             "vocabulary": {"tokens": ["<PAD>", "a", "b"]},
             "config": {"context_length": 256}}
    second = copy.deepcopy(first)
    second["config"]["context_length"] = 512
    return [first, second]


def test_longer_model_is_compared_at_shared_context(bundles):
    comparison.validate_comparison_contract(bundles, 256)
    with pytest.raises(ValueError, match="position capacity"):
        comparison.validate_comparison_contract(bundles, 512)


def test_same_size_different_data_and_encoding_cannot_be_ranked(bundles):
    changed = copy.deepcopy(bundles)
    changed[1]["summary"]["manifest_sha256"] = "different-rows-same-counts"
    with pytest.raises(ValueError, match="data manifest"):
        comparison.validate_comparison_contract(changed, 256)
    changed = copy.deepcopy(bundles)
    changed[1]["vocabulary"]["tokens"] = ["<PAD>", "b", "a"]
    with pytest.raises(ValueError, match="encodings"):
        comparison.validate_comparison_contract(changed, 256)


def test_comparison_cannot_overwrite_training_evidence(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="immutable training"):
        comparison.compare([run], run / "comparison.json", device="cpu", root=tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("original")
    with pytest.raises(FileExistsError):
        comparison.compare([run], receipt, device="cpu", root=tmp_path)
    assert receipt.read_text() == "original"
