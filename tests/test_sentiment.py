import json
from pathlib import Path

import numpy as np
import pytest
from lab1.common import task_config_path
import torch

from lab1 import sentiment


@pytest.fixture
def config():
    return json.loads(task_config_path("sentiment").read_text())["smoke"]


def test_preprocessing_preserves_negation_and_training_only_vocabulary():
    tokens = sentiment.tokenize("<b>I didn't like it</b>; it's NOT good, never again!")
    assert "not" in tokens and "never" in tokens
    assert "didn" not in tokens and "b" not in tokens
    assert "the" not in sentiment.tokenize("the cold food")
    vocabulary = sentiment.build_vocabulary(["warm good food", "good kind staff"], 100)
    assert "heldoutword" not in vocabulary
    encoded = sentiment.Reviews([{"id": "validation:0", "text": "heldoutword", "label": 0}], vocabulary, 32)
    assert encoded[0][0].tolist() == [sentiment.UNK]
    assert encoded.oov_rates == [1.0]


@pytest.mark.parametrize("name", sentiment.MODEL_NAMES)
def test_padding_does_not_change_logits(name, config):
    torch.manual_seed(2342)
    torch.set_num_threads(2)
    model = sentiment.build_model(name, 20, config).eval()
    short = torch.tensor([[2, 3, 4]])
    padded = torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0]])
    with torch.inference_mode():
        assert torch.allclose(model(short), model(padded), atol=1e-6, rtol=1e-5)
        assert torch.isfinite(model(torch.tensor([[1, 0, 0]]))).all()


def test_official_model_shapes_and_independent_embeddings(config):
    full = json.loads(task_config_path("sentiment").read_text())["full"]
    models = [sentiment.build_model(name, 100, full) for name in sentiment.MODEL_NAMES]
    assert all(model.embedding.embedding_dim == 128 for model in models)
    assert models[1].encoder.hidden_size == 96 and models[1].encoder.bidirectional
    assert [block.first.dilation[0] for block in models[2].blocks] == [1, 2, 4, 8]
    assert len({model.embedding.weight.data_ptr() for model in models}) == 3


def test_metrics_probabilities_confusion_and_pairing():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    metrics, curves = sentiment.compute_metrics(y, probability, bootstrap_samples=20)
    assert metrics["accuracy"] == metrics["macro"]["f1"] == metrics["mcc"] == 1.0
    assert metrics["roc_auc"] == metrics["pr_auc_trapezoidal"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["brier"] == pytest.approx(0.025)
    assert metrics["ece_15"] == pytest.approx(0.15)
    assert sum(row["count"] for row in curves["reliability"]) == len(y)
    compared = sentiment.paired_mcnemar(y, probability, np.array([0.9, 0.8, 0.8, 0.9]))
    assert compared["table"] == [[2, 2], [0, 0]]
    assert compared["pvalue_exact_two_sided"] == pytest.approx(0.5)
    assert sentiment.paired_mcnemar(y, probability, probability)["pvalue_exact_two_sided"] == 1.0
    with pytest.raises(ValueError):
        sentiment.compute_metrics(y, np.array([0.1, 0.2, np.nan, 0.8]))


def test_split_overlap_is_rejected():
    row = {"id": "shared", "text": "hello", "label": 0}
    with pytest.raises(ValueError, match="overlap"):
        sentiment.audit_splits({"train": [row], "validation": [row], "test": [{**row, "id": "test"}]})


def test_empty_reviews_get_unknown_token(config):
    dataset = sentiment.Reviews([{"id": "empty", "text": "the and it", "label": 0}], {"<pad>": 0, "<unk>": 1}, 32)
    ids, labels, indices = sentiment.collate_reviews([dataset[0]])
    assert ids.tolist() == [[1]] and labels.tolist() == [0.0] and indices.tolist() == [0]


def test_synthetic_suite_checkpoint_reload_and_resume(tmp_path, config, monkeypatch):
    # No datasets package/network path is used in smoke mode.
    result = sentiment.run(config, tmp_path, "cpu")
    assert result["synthetic"] is True and set(result["models"]) == set(sentiment.MODEL_NAMES)
    predictions = {name: (tmp_path / name / "test_predictions.csv").read_text() for name in sentiment.MODEL_NAMES}
    for name in sentiment.MODEL_NAMES:
        checkpoint = torch.load(tmp_path / name / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
        assert {"optimizer", "rng", "vocabulary", "loader_rng", "scaler", "scheduler"} <= checkpoint.keys()
        restored = sentiment.build_model(name, len(checkpoint["vocabulary"]), checkpoint["config"])
        restored.load_state_dict(checkpoint["model"])
        assert (tmp_path / name / "learning_curves.png").is_file()
    with pytest.raises(FileExistsError):
        sentiment.run(config, tmp_path, "cpu")
    sentiment.run(config, tmp_path, "cpu", resume=tmp_path)
    assert all((tmp_path / name / "test_predictions.csv").read_text() == predictions[name] for name in sentiment.MODEL_NAMES)
    assert json.loads((tmp_path / "summary.json").read_text())["manual_error_review_complete"] is False
    changed = {**config, "seed": 999}
    preserved = {p: p.read_bytes() for p in tmp_path.glob("*") if p.is_file()}
    with pytest.raises(ValueError, match="fingerprint"):
        sentiment.run(changed, tmp_path, "cpu", resume=tmp_path)
    assert all(p.read_bytes() == contents for p, contents in preserved.items())
    with pytest.raises(ValueError, match="fingerprint"):
        sentiment.run(changed, tmp_path / "bad_resume", "cpu", resume=tmp_path)
