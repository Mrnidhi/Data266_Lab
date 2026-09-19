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


def test_preencoded_cache_matches_raw_and_rejects_stale_data(tmp_path, config):
    folder = tmp_path / "features"
    sentiment.prepare_features(config, folder)
    cached_config = dict(config, encoded_cache=str(folder))
    records = sentiment._load_records(config)
    vocabulary, datasets = sentiment._load_features(cached_config, records)
    assert vocabulary == sentiment.build_vocabulary((r["text"] for r in records["train"]),
                                                    config["vocabulary_size"], config["min_frequency"])
    for name, rows in records.items():
        original = sentiment.Reviews(rows, vocabulary, config["max_length"])
        for actual, expected in zip(datasets[name].encoded, original.encoded):
            np.testing.assert_array_equal(actual, expected)
        for key in ("lengths", "oov_rates", "negations"):
            assert getattr(datasets[name], key) == getattr(original, key)
    changed_records = {name: [dict(r) for r in rows] for name, rows in records.items()}
    changed_records["train"][0]["text"] += " changed"
    with pytest.raises(ValueError, match="fingerprint"):
        sentiment._load_features(cached_config, changed_records)
    with pytest.raises(ValueError, match="fingerprint"):
        sentiment._load_features(dict(cached_config, max_length=16), records)
    path = folder / "train.npz"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        sentiment._load_features(cached_config, records)


def test_cpu_lengths_preserve_lstm_outputs_and_gradients(config):
    torch.manual_seed(2342)
    model = sentiment.build_model("bilstm", 20, config).eval()
    ids = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
    expected = model(ids)
    expected.sum().backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    actual = sentiment._forward(model, ids, "cpu")
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for p, expected_gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(p.grad, expected_gradient, rtol=0, atol=0)


def test_validation_only_candidate_cannot_access_test_split(tmp_path, config):
    cfg = dict(config, validation_only=True)
    records = sentiment._load_records(cfg)
    vocabulary = sentiment.build_vocabulary((r["text"] for r in records["train"]), 128)
    datasets = {name: sentiment.Reviews(records[name], vocabulary, cfg["max_length"])
                for name in ("train", "validation")}
    metrics, y, probability = sentiment._train_one("maxpool_mlp", cfg, datasets, vocabulary,
                                                 tmp_path, "cpu", "test-only", None)
    assert metrics["test_evaluated"] is False and y is None and probability is None
    assert (tmp_path / "maxpool_mlp/checkpoints/best.pt").exists()
    assert not (tmp_path / "maxpool_mlp/test_predictions.csv").exists()


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


def test_empty_scaler_and_fewer_cuda_devices_can_resume(monkeypatch):
    class EnabledScaler:
        def is_enabled(self):
            return True

        def load_state_dict(self, state):
            assert state
            self.loaded = state

    scaler = EnabledScaler()
    assert sentiment._restore_scaler(scaler, {}) == "fresh_scaler_no_saved_fp16_state"
    assert sentiment._restore_scaler(scaler, {"scale": 32}) == "restored"
    assert scaler.loaded == {"scale": 32}
    state = sentiment._rng_state()
    state["cuda"] = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, index: calls.append((value.item(), index)))
    sentiment._restore_rng(state)
    assert calls == [(1, 0)]


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("interruption", ["first_batch", "before_validation", "legacy_epoch"])
def test_interrupted_training_is_exact_and_portable(tmp_path, config, monkeypatch, interruption):
    config = dict(config, epochs=2, checkpoint_every_steps=1, early_stopping_patience=4, cpu_threads=1)
    baseline, partial, relocated = (tmp_path / name for name in ("baseline", "partial", "relocated"))
    sentiment.run(config, baseline, "cpu")
    original_save = sentiment._atomic_checkpoint

    class Interrupted(Exception):
        pass

    def interrupt(path, state):
        progress = state.get("epoch_progress")
        should_stop = path.name == "last.pt" and state["model_name"] == "maxpool_mlp" and (
            (interruption == "first_batch" and progress and progress["next_batch"] == 1)
            or (interruption == "before_validation" and progress and progress["next_batch"] == 3)
            or (interruption == "legacy_epoch" and state["epoch"] == 1 and not progress))
        if should_stop and interruption == "legacy_epoch":
            # Reproduce the old epoch-only schema and fingerprint for both saved files.
            records = sentiment._load_records(config)
            for target, value in ((path, state), (path.parent / "best.pt", torch.load(path.parent / "best.pt", weights_only=False))):
                value = dict(value)
                value["format_version"] = 1
                value.pop("epoch_progress", None)
                value.pop("global_step", None)
                value["fingerprint"] = sentiment._fingerprint(config, value["vocabulary"], records, legacy=True)
                original_save(target, value)
        else:
            original_save(path, state)
        if should_stop:
            raise Interrupted()

    monkeypatch.setattr(sentiment, "_atomic_checkpoint", interrupt)
    with pytest.raises(Interrupted):
        sentiment.run(config, partial, "cpu")
    monkeypatch.setattr(sentiment, "_atomic_checkpoint", original_save)
    settings = dict(config, raw_data_cache=str(tmp_path / "new_machine_cache"), checkpoint_every_steps=2,
                    log_every_steps=1)
    sentiment.run(settings, relocated, "cpu", resume=partial)
    for name in sentiment.MODEL_NAMES:
        left = torch.load(baseline / name / "checkpoints" / "last.pt", weights_only=False)
        right = torch.load(relocated / name / "checkpoints" / "last.pt", weights_only=False)
        for key in ("model", "optimizer", "scheduler", "rng", "loader_rng", "best_f1", "bad_epochs", "global_step", "epoch"):
            _assert_nested_equal(left[key], right[key])
        for a, b in zip(left["history"], right["history"]):
            for key in ("epoch", "train_loss", "validation_loss", "validation_macro_f1", "learning_rate"):
                assert a[key] == b[key]
        assert (baseline / name / "test_predictions.csv").read_bytes() == (relocated / name / "test_predictions.csv").read_bytes()
