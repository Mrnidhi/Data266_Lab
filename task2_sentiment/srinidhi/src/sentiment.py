"""Scratch-trained Yelp classifiers and reproducible, paired evaluation.

The synthetic smoke mode is an integration check, never a Yelp result.
Downloads occur only when a caller explicitly runs rehearsal/full mode.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import random
import re
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


PAD, UNK = 0, 1
# A frozen, task-specific stopword list. Negation and contrast words stay intact.
STOPWORDS = frozenset("a an the and or of to in on at for from with by as is am are was were be been being this that these those it its i me my we us our you your he him his she her they them their have has had do does did would could should will shall can may might".split())
MODEL_NAMES = ("maxpool_mlp", "bilstm", "dilated_cnn")


def tokenize(text: str) -> list[str]:
    text = html.unescape(text).casefold().replace("’", "'")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+", " url ", text)
    for original, expanded in (("won't", "will not"), ("can't", "can not"), ("shan't", "shall not")):
        text = text.replace(original, expanded)
    text = re.sub(r"n['’]t\b", " not", text)
    for suffix, expansion in (("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'m", " am"), ("'d", " would"), ("'s", " is")):
        text = text.replace(suffix, expansion)
    return [word for word in re.findall(r"\b\w+\b", text, flags=re.UNICODE) if word not in STOPWORDS]


def build_vocabulary(texts, size: int, min_frequency: int = 1) -> dict[str, int]:
    counts = Counter(word for text in texts for word in tokenize(text))
    words = sorted(counts, key=lambda word: (-counts[word], word))
    words = [word for word in words if counts[word] >= min_frequency][:max(0, size - 2)]
    return {"<pad>": PAD, "<unk>": UNK, **{word: i + 2 for i, word in enumerate(words)}}


class Reviews(Dataset):
    def __init__(self, records: list[dict], vocabulary: dict[str, int], max_length: int):
        self.records = records
        self.encoded = []
        self.lengths, self.oov_rates, self.negations = [], [], []
        for row in records:
            tokens = tokenize(row["text"])
            self.lengths.append(len(tokens))
            self.oov_rates.append(sum(t not in vocabulary for t in tokens) / max(1, len(tokens)))
            self.negations.append(any(t in {"not", "no", "never", "neither", "nor", "without"} for t in tokens))
            ids = [vocabulary.get(word, UNK) for word in tokens[:max_length]] or [UNK]
            self.encoded.append(np.asarray(ids, dtype=np.int32))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.encoded[index], self.records[index]["label"], index


def collate_reviews(batch):
    width = max(len(row[0]) for row in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (tokens, _, _) in enumerate(batch):
        ids[i, :len(tokens)] = torch.as_tensor(tokens, dtype=torch.long)
    return ids, torch.tensor([row[1] for row in batch], dtype=torch.float32), torch.tensor([row[2] for row in batch])


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pooled = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min).amax(dim=1)
    return torch.where(mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))


class MaxPoolMLP(nn.Module):
    def __init__(self, vocabulary_size: int, cfg: dict):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, cfg["embedding_dim"], padding_idx=PAD)
        self.head = nn.Sequential(nn.Linear(cfg["embedding_dim"], cfg["mlp_hidden"]), nn.ReLU(), nn.Dropout(cfg["dropout"]), nn.Linear(cfg["mlp_hidden"], 1))

    def forward(self, ids):
        return self.head(masked_max(self.embedding(ids), ids.ne(PAD))).squeeze(-1)


class BiLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, cfg: dict):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, cfg["embedding_dim"], padding_idx=PAD)
        self.embedding_dropout = nn.Dropout(cfg["dropout"])
        self.encoder = nn.LSTM(cfg["embedding_dim"], cfg["lstm_hidden"], num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * cfg["lstm_hidden"], cfg["mlp_hidden"]), nn.ReLU(), nn.Dropout(cfg["dropout"]), nn.Linear(cfg["mlp_hidden"], 1))

    def forward(self, ids):
        mask = ids.ne(PAD)
        lengths = mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(self.embedding_dropout(self.embedding(ids)), lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        values, _ = pad_packed_sequence(encoded, batch_first=True, total_length=ids.shape[1])
        return self.head(masked_max(values, mask)).squeeze(-1)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.first = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask):
        # Mask after each convolution, so padding cannot flow back into real tokens.
        hidden = self.dropout(torch.relu(self.first(values))) * mask
        hidden = self.dropout(torch.relu(self.second(hidden))) * mask
        return torch.relu(values + hidden) * mask


class DilatedCNN(nn.Module):
    def __init__(self, vocabulary_size: int, cfg: dict):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, cfg["embedding_dim"], padding_idx=PAD)
        self.projection = nn.Conv1d(cfg["embedding_dim"], cfg["cnn_channels"], 1)
        self.blocks = nn.ModuleList(ResidualDilatedBlock(cfg["cnn_channels"], dilation, cfg["cnn_dropout"]) for dilation in cfg["dilations"])
        self.head = nn.Sequential(nn.Linear(cfg["cnn_channels"], cfg["mlp_hidden"]), nn.ReLU(), nn.Dropout(cfg["dropout"]), nn.Linear(cfg["mlp_hidden"], 1))

    def forward(self, ids):
        mask = ids.ne(PAD)
        channels_mask = mask.unsqueeze(1)
        hidden = self.projection(self.embedding(ids).transpose(1, 2)) * channels_mask
        for block in self.blocks:
            hidden = block(hidden, channels_mask)
        return self.head(masked_max(hidden.transpose(1, 2), mask)).squeeze(-1)


def build_model(name: str, vocabulary_size: int, cfg: dict) -> nn.Module:
    return {"maxpool_mlp": MaxPoolMLP, "bilstm": BiLSTM, "dilated_cnn": DilatedCNN}[name](vocabulary_size, cfg)


def _dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _csv(path: Path, rows: list[dict], columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        for index, rng in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng.cpu(), index)


def _restore_scaler(scaler, saved_state):
    if not scaler.is_enabled():
        return "disabled_for_current_device_precision"
    if saved_state:
        scaler.load_state_dict(saved_state)
        return "restored"
    # BF16/CPU checkpoints have no FP16 loss scale; initialize it on the new GPU.
    return "fresh_scaler_no_saved_fp16_state"


def _resume_contract(config):
    local_settings = {"data_dir", "raw_data_cache", "offline", "num_workers", "cpu_threads",
                      "checkpoint_every_steps", "log_every_steps"}
    return {key: value for key, value in config.items() if key not in local_settings}


def _fingerprint(config, vocabulary, records, legacy=False):
    contract = {key: value for key, value in config.items() if key not in {"data_dir", "log_every_steps"}} if legacy else _resume_contract(config)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode() + json.dumps(vocabulary, sort_keys=True).encode())
    for name in ("train", "validation", "test"):
        for row in records[name]:
            digest.update(json.dumps(row, sort_keys=True).encode())
    return digest.hexdigest()


def _synthetic_records(count, offset, split):
    positive = ["Wonderful friendly service and excellent tasty food", "I never had a bad meal and the staff were lovely", "Not disappointing: delicious food and warm service", "Excellent value, I will happily return"]
    negative = ["Terrible cold food with rude service", "I will never return: horrible and disappointing", "Not good at all, the meal was awful", "Bad value, slow service, and unpleasant staff"]
    return [{"id": f"synthetic:{split}:{i}", "text": (positive if i % 2 else negative)[(i // 2 + offset) % 4] + f" visit {i + offset}", "label": i % 2} for i in range(count)]


def _load_records(cfg: dict):
    if cfg["mode"] == "smoke":
        return {name: _synthetic_records(cfg[f"{name}_limit"], j, name) for j, name in enumerate(("train", "validation", "test"))}
    if cfg.get("data_dir"):
        folder = Path(cfg["data_dir"])
        records = {name: [json.loads(line) for line in (folder / f"{name}.jsonl").read_text().splitlines() if line.strip()] for name in ("train", "validation", "test")}
        for name, rows in records.items():
            if not rows or any(row.get("label") not in (0, 1) or not isinstance(row.get("text"), str) or not isinstance(row.get("id"), str) for row in rows):
                raise ValueError(f"Invalid local {name} records")
        expected = {"train": 504000, "validation": 56000, "test": 38000} if cfg["mode"] == "full" else {name: cfg[f"{name}_limit"] for name in records}
        if any(expected[name] is not None and len(rows) != expected[name] for name, rows in records.items()):
            raise ValueError("Local dataset counts do not match the selected mode")
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["seed"] != cfg["seed"] or manifest["mode"] != cfg["mode"]:
            raise ValueError("Local dataset mode/seed differs from configuration")
        if any(hashlib.sha256((folder / f"{name}.jsonl").read_bytes()).hexdigest() != manifest["sha256"][name] for name in records):
            raise ValueError("Local dataset checksum mismatch")
        return records
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    data = load_dataset(cfg["dataset"], revision=cfg.get("dataset_revision", "main"),
                        cache_dir=cfg.get("raw_data_cache"))
    if len(data["train"]) != 560000 or len(data["test"]) != 38000:
        raise ValueError("Expected official Yelp Polarity splits of 560000 and 38000 rows")
    labels = np.asarray(data["train"]["label"])
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected Hugging Face Yelp polarity labels 0=negative, 1=positive")
    training, validation = train_test_split(np.arange(len(labels)), test_size=0.1, stratify=labels, random_state=cfg["seed"])
    def select(indices, all_labels, limit):
        if limit is None or limit >= len(indices):
            return np.sort(indices)
        selected, _ = train_test_split(indices, train_size=limit, stratify=all_labels[indices], random_state=cfg["seed"])
        return np.sort(selected)
    training = select(training, labels, cfg["train_limit"])
    validation = select(validation, labels, cfg["validation_limit"])
    test_labels = np.asarray(data["test"]["label"])
    if set(np.unique(test_labels)) != {0, 1}:
        raise ValueError("Unexpected test labels")
    test = select(np.arange(len(test_labels)), test_labels, cfg["test_limit"])
    def rows(split, indices):
        return [{"id": f"yelp:{split}:{int(index)}", "text": row["text"], "label": int(row["label"])} for index, row in zip(indices, data[split].select(indices.tolist()))]
    return {"train": rows("train", training), "validation": rows("train", validation), "test": rows("test", test)}


def prefetch_data(config: dict, output_dir: Path) -> dict:
    """Download/select official rows locally; returned folder is usable as data_dir.

    This fetches the official dataset once (not a model) and materializes only the
    selected split rows. Calling it is explicit; importing this module is offline.
    """
    cfg = {**config, "data_dir": None}
    if cfg["mode"] == "smoke":
        raise ValueError("Synthetic smoke data does not need prefetching")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        _load_records({**config, "data_dir": str(output_dir)})
        return json.loads((output_dir / "manifest.json").read_text())
    records = _load_records(cfg)
    manifest = {"mode": cfg["mode"], "seed": cfg["seed"], "dataset": cfg["dataset"], "requested_revision": cfg.get("dataset_revision", "main"), "split_counts": {}, "sha256": {}, "data_dir": str(output_dir)}
    for name, rows in records.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        manifest["split_counts"][name] = len(rows)
        manifest["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _dump(output_dir / "manifest.json", manifest)
    return manifest


def audit_splits(records):
    ids = {name: {row["id"] for row in rows} for name, rows in records.items()}
    if ids["train"] & (ids["validation"] | ids["test"]) or ids["validation"] & ids["test"]:
        raise ValueError("Split IDs overlap")
    hashes = {name: Counter(hashlib.sha256(" ".join(row["text"].casefold().split()).encode()).hexdigest() for row in rows) for name, rows in records.items()}
    return {"split_counts": {name: len(rows) for name, rows in records.items()}, "within_split_duplicate_rows": {name: sum(n - 1 for n in counts.values()) for name, counts in hashes.items()}, "cross_split_shared_text_hashes": {f"{a}__{b}": len(hashes[a].keys() & hashes[b].keys()) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}, "duplicate_policy": "Audit only; official test rows are preserved unchanged.", "label_mapping": {"0": "negative", "1": "positive"}}


def reliability_bins(y, probabilities, bins=15):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = (probabilities >= 0.5) == y
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        chosen = assignments == index
        count = int(chosen.sum())
        accuracy = float(correct[chosen].mean()) if count else None
        mean_confidence = float(confidence[chosen].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - mean_confidence)
        rows.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins, "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    return float(ece), rows


def _bootstrap_metrics(y, predicted, repeats, seed):
    rng = np.random.default_rng(seed)
    values = np.empty((repeats, 3), dtype=np.float64)
    for index in range(repeats):
        chosen = rng.integers(0, len(y), len(y))
        tn, fp, fn, tp = np.bincount(2 * y[chosen] + predicted[chosen], minlength=4).astype(float)
        accuracy = (tp + tn) / len(y)
        positive_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        negative_f1 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
        values[index] = accuracy, (positive_f1 + negative_f1) / 2, mcc
    return {name: {"low": float(np.quantile(values[:, i], 0.025)), "high": float(np.quantile(values[:, i], 0.975))} for i, name in enumerate(("accuracy", "macro_f1", "mcc"))}


def compute_metrics(y, probabilities, bootstrap_samples=0, seed=2342):
    from sklearn.metrics import accuracy_score, auc, brier_score_loss, confusion_matrix, matthews_corrcoef, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(y) != len(probabilities) or len(y) == 0 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Metrics require equal, nonempty arrays with finite probabilities in [0,1]")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Only binary labels 0 and 1 are supported")
    predicted = (probabilities >= 0.5).astype(np.int64)
    result = {"count": len(y), "accuracy": float(accuracy_score(y, predicted)), "mcc": float(matthews_corrcoef(y, predicted)), "brier": float(brier_score_loss(y, probabilities)), "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(), "threshold": 0.5}
    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, labels=[0, 1], average=average, zero_division=0)
        result[average] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=[0, 1], zero_division=0)
    result["per_class"] = [{"label": i, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i in (0, 1)]
    result["ece_15"], calibration = reliability_bins(y, probabilities, 15)
    result["ece_definition"] = "Top-label confidence ECE; 15 equal-width bins on [0,1]; threshold 0.5."
    curves = {"reliability": calibration}
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, probabilities)
        pr_precision, pr_recall, _ = precision_recall_curve(y, probabilities)
        result["roc_auc"] = float(roc_auc_score(y, probabilities))
        result["pr_auc_trapezoidal"] = float(auc(pr_recall, pr_precision))
        curves.update(roc={"fpr": fpr.tolist(), "tpr": tpr.tolist()}, precision_recall={"recall": pr_recall.tolist(), "precision": pr_precision.tolist()})
    else:
        result.update(roc_auc=None, pr_auc_trapezoidal=None)
    if bootstrap_samples:
        result["bootstrap_95_percentile"] = _bootstrap_metrics(y, predicted, bootstrap_samples, seed)
        result["bootstrap_samples"] = bootstrap_samples
        result["uncertainty_scope"] = "IID test-row resampling; does not include training-seed uncertainty."
    return result, curves


def paired_mcnemar(y, baseline_probabilities, other_probabilities):
    from scipy.stats import binomtest
    y = np.asarray(y)
    baseline_correct = (np.asarray(baseline_probabilities) >= 0.5) == y
    other_correct = (np.asarray(other_probabilities) >= 0.5) == y
    a = int((baseline_correct & other_correct).sum())
    b = int((baseline_correct & ~other_correct).sum())
    c = int((~baseline_correct & other_correct).sum())
    d = int((~baseline_correct & ~other_correct).sum())
    return {"table": [[a, b], [c, d]], "table_definition": "Rows baseline correct/incorrect, columns other correct/incorrect", "discordant_pairs": b + c, "pvalue_exact_two_sided": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0, "test": "Paired exact McNemar (binomial), no continuity correction", "multiple_comparisons": "Two prespecified baseline comparisons; p-values are unadjusted."}


def _slice_metrics(dataset, y, probabilities):
    lengths, oov, negation = np.asarray(dataset.lengths), np.asarray(dataset.oov_rates), np.asarray(dataset.negations)
    masks = {"length_lt50": lengths < 50, "length_50_199": (lengths >= 50) & (lengths < 200), "length_ge200": lengths >= 200, "negation_present": negation, "negation_absent": ~negation, "oov_lt10pct": oov < 0.1, "oov_ge10pct": oov >= 0.1}
    return {name: compute_metrics(y[mask], probabilities[mask])[0] if mask.any() else {"count": 0} for name, mask in masks.items()}


def _loader(dataset, batch_size, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0, collate_fn=collate_reviews)


def _predict(model, loader, device):
    model.eval()
    probabilities, labels, indices, loss_sum = [], [], [], 0.0
    with torch.inference_mode():
        for ids, y, index in loader:
            logits = model(ids.to(device))
            loss_sum += nn.functional.binary_cross_entropy_with_logits(logits.float(), y.to(device), reduction="sum").item()
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels.append(y.numpy().astype(np.int64))
            indices.append(index.numpy())
    probabilities, labels, indices = map(np.concatenate, (probabilities, labels, indices))
    order = np.argsort(indices)
    return labels[order], probabilities[order], loss_sum / len(labels)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def _atomic_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _plot(model_dir, history, metrics, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for key in ("train_loss", "validation_loss"):
        axes[0].plot([r["epoch"] for r in history], [r[key] for r in history], marker="o", label=key)
    axes[0].set(xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[0].legend()
    axes[1].plot([r["epoch"] for r in history], [r["validation_macro_f1"] for r in history], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro F1", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(model_dir / "learning_curves.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    if "roc" in curves:
        axes[0].plot(curves["roc"]["fpr"], curves["roc"]["tpr"])
        axes[1].plot(curves["precision_recall"]["recall"], curves["precision_recall"]["precision"])
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    bins = [row for row in curves["reliability"] if row["count"]]
    axes[2].plot([row["confidence"] for row in bins], [row["accuracy"] for row in bins], marker="o")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set(xlabel="Confidence", ylabel="Accuracy", title="Reliability", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout(); fig.savefig(model_dir / "evaluation_curves.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    matrix = np.asarray(metrics["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label", title="Confusion matrix")
    fig.tight_layout(); fig.savefig(model_dir / "confusion_matrix.png", dpi=150); plt.close(fig)


def _train_one(name, cfg, datasets, vocabulary, output, device, fingerprint, resume, accepted_fingerprints=None):
    _seed(cfg["seed"])
    model = build_model(name, len(vocabulary), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rates"][name], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    cuda = str(device).startswith("cuda")
    amp_dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and amp_dtype == torch.float16 and cfg["amp"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    model_dir = output / name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    history, epoch_start, best_f1, bad_epochs = [], 0, -1.0, 0
    stopped = False
    epoch_progress, global_step = None, 0
    resume_scaler_status = "not_resumed"
    checkpoint_every = cfg.get("checkpoint_every_steps", 250)
    if not isinstance(checkpoint_every, int) or checkpoint_every < 1:
        raise ValueError("checkpoint_every_steps must be a positive integer")
    log_every = cfg.get("log_every_steps", 50)
    if not isinstance(log_every, int) or log_every < 1:
        raise ValueError("log_every_steps must be a positive integer")
    if resume:
        source = resume / name / "checkpoints" / "last.pt" if resume.is_dir() else resume
        if source.exists():
            state = torch.load(source, map_location="cpu", weights_only=False)
            if state["model_name"] != name:
                raise ValueError("Resume a suite directory, or use a checkpoint matching the requested model")
            if state["fingerprint"] not in (accepted_fingerprints or {fingerprint}):
                raise ValueError("Resume configuration/data fingerprint mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor) and key != "step":
                        optimizer_state[key] = value.to(device)
            scheduler.load_state_dict(state["scheduler"])
            resume_scaler_status = _restore_scaler(scaler, state["scaler"])
            generator.set_state(state["loader_rng"].cpu())
            _restore_rng(state["rng"])
            history, epoch_start, best_f1, bad_epochs = state["history"], state["epoch"], state["best_f1"], state["bad_epochs"]
            stopped = state["stopped_early"]
            epoch_progress = state.get("epoch_progress")
            batches_per_epoch = (len(datasets["train"]) + cfg["batch_size"] - 1) // cfg["batch_size"]
            global_step = state.get("global_step", epoch_start * batches_per_epoch)
            # Allow outputs to be written to a fresh folder when resuming elsewhere.
            import shutil
            prior_best = source.parent / "best.pt"
            if prior_best.exists() and prior_best.resolve() != (checkpoint_dir / "best.pt").resolve():
                shutil.copy2(prior_best, checkpoint_dir / "best.pt")
            if source.resolve() != (checkpoint_dir / "last.pt").resolve():
                shutil.copy2(source, checkpoint_dir / "last.pt")

    def checkpoint_state(completed_epochs, progress=None):
        return {"format_version": 2, "model_name": name, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng": _rng_state(), "loader_rng": generator.get_state(),
                "vocabulary": vocabulary, "config": cfg, "fingerprint": fingerprint,
                "epoch": completed_epochs, "epoch_progress": progress, "global_step": global_step,
                "history": history, "best_f1": best_f1, "bad_epochs": bad_epochs, "stopped_early": stopped}

    validation_loader = _loader(datasets["validation"], cfg["batch_size"])
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    batches_per_epoch = (len(datasets["train"]) + cfg["batch_size"] - 1) // cfg["batch_size"]
    planned_steps = batches_per_epoch * cfg["epochs"]
    session_started, session_initial_step = time.perf_counter(), global_step
    print(f"sentiment/{name} mode={cfg['mode']} device={device} step={global_step}/{planned_steps}; "
          "ETA estimates assume all planned epochs; early stopping may shorten training and evaluation/export may add time", flush=True)
    for epoch in range(epoch_start, cfg["epochs"]):
        if stopped:
            break
        progress = epoch_progress or {}
        if progress:
            generator.set_state(progress["epoch_start_loader_rng"].cpu())
        epoch_start_loader_rng = generator.get_state().clone()
        train_loader = _loader(datasets["train"], cfg["batch_size"], True, generator)
        next_batch = progress.get("next_batch", 0)
        if not 0 <= next_batch <= len(train_loader):
            raise ValueError("Saved sentiment batch position is outside the epoch")
        model.train()
        _sync(device); started = time.perf_counter()
        train_loss, count = progress.get("train_loss", 0.0), progress.get("count", 0)
        prior_train_seconds = progress.get("train_seconds", 0.0)
        for batch_index, (ids, y, _) in enumerate(train_loader):
            # Replay only the deterministic sampler/collation, never completed updates.
            if batch_index < next_batch:
                continue
            ids, y = ids.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=amp_dtype) if cuda and cfg["amp"] else nullcontext()
            with context:
                logits = model(ids)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in {name}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item() * len(y)
            count += len(y)
            global_step += 1
            if global_step % log_every == 0 or batch_index + 1 == len(train_loader):
                elapsed = time.perf_counter() - session_started
                completed = global_step - session_initial_step
                eta = elapsed / completed * max(0, planned_steps - global_step)
                print(f"sentiment/{name} epoch={epoch + 1}/{cfg['epochs']} "
                      f"batch={batch_index + 1}/{len(train_loader)} step={global_step}/{planned_steps} "
                      f"loss={loss.item():.4f} mean_loss={train_loss/count:.4f} "
                      f"elapsed={elapsed:.1f}s eta_steps_est={eta:.1f}s", flush=True)
            if global_step % checkpoint_every == 0:
                _sync(device)
                progress = {"next_batch": batch_index + 1, "epoch_start_loader_rng": epoch_start_loader_rng,
                            "train_loss": train_loss, "count": count,
                            "train_seconds": prior_train_seconds + time.perf_counter() - started}
                _atomic_checkpoint(checkpoint_dir / "last.pt", checkpoint_state(epoch, progress))
        _sync(device); train_seconds = prior_train_seconds + time.perf_counter() - started
        print(f"sentiment/{name} epoch={epoch + 1} validating", flush=True)
        valid_y, valid_probability, valid_loss = _predict(model, validation_loader, device)
        validation_metrics, _ = compute_metrics(valid_y, valid_probability)
        f1 = validation_metrics["macro"]["f1"]
        scheduler.step(valid_loss)
        improved = f1 > best_f1 + cfg["early_stopping_min_delta"]
        best_f1, bad_epochs = (f1, 0) if improved else (best_f1, bad_epochs + 1)
        stopped = bad_epochs >= cfg["early_stopping_patience"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss / count, "validation_loss": valid_loss, "validation_macro_f1": f1, "train_seconds": train_seconds, "examples_per_second": count / max(train_seconds, 1e-9), "learning_rate": optimizer.param_groups[0]["lr"]})
        epoch_progress = None
        state = checkpoint_state(epoch + 1)
        if improved:
            _atomic_checkpoint(checkpoint_dir / "best.pt", state)
        _atomic_checkpoint(checkpoint_dir / "last.pt", state)
        _dump(model_dir / "history.json", history)
        print(f"sentiment/{name} epoch={epoch + 1} train_loss={train_loss/count:.4f} val_macro_f1={f1:.4f}", flush=True)
    best = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    y, probabilities, test_loss = _predict(model, _loader(datasets["test"], cfg["batch_size"]), device)
    metrics, curves = compute_metrics(y, probabilities, cfg["bootstrap_samples"], cfg["seed"])
    metrics.update(test_loss=test_loss, selected_epoch=best["epoch"], selected_validation_macro_f1=best["best_f1"], parameter_count=sum(p.numel() for p in model.parameters()), train_seconds=sum(row["train_seconds"] for row in history), peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if cuda else None, mode=cfg["mode"], synthetic=cfg["mode"] == "smoke", resume_scaler_status=resume_scaler_status, completed_training_steps=global_step)
    _dump(model_dir / "metrics.json", metrics)
    _dump(model_dir / "curves.json", curves)
    _dump(model_dir / "slices.json", _slice_metrics(datasets["test"], y, probabilities))
    rows = [{"example_id": row["id"], "true_label": int(y[i]), "predicted_label": int(probabilities[i] >= 0.5), "positive_probability": float(probabilities[i]), "original_token_length": datasets["test"].lengths[i], "oov_rate": datasets["test"].oov_rates[i], "has_negation": datasets["test"].negations[i]} for i, row in enumerate(datasets["test"].records)]
    _csv(model_dir / "test_predictions.csv", rows)
    errors = np.flatnonzero((probabilities >= 0.5) != y)
    errors = sorted(errors, key=lambda i: (-max(probabilities[i], 1 - probabilities[i]), datasets["test"].records[i]["id"]))[:20]
    error_rows = [{**rows[i], "text": datasets["test"].records[i]["text"], "manual_group": "", "manual_explanation": "", "reviewed_by": ""} for i in errors]
    error_columns = list(rows[0]) + ["text", "manual_group", "manual_explanation", "reviewed_by"]
    _csv(model_dir / "error_candidates_for_manual_review.csv", error_rows, error_columns)
    _dump(model_dir / "error_review_instructions.json", {"required_review_count": 20, "available_error_candidates": len(error_rows), "status": "Manual interpretation required; no explanations or group assignments have been fabricated.", "suggested_groups": ["negation", "sarcasm", "mixed sentiment", "truncation or missed context", "ambiguous or noisy label"], "selection": "Up to 20 most confidently incorrect test predictions; list does not represent an unbiased sample of errors."})
    _plot(model_dir, history, metrics, curves)
    del model, optimizer
    if cuda:
        torch.cuda.empty_cache()
    return metrics, y, probabilities


def run(config: dict, output_dir: Path, device: str, resume: Path | None = None) -> dict:
    """Train/evaluate all three models; resume saved batches or legacy epoch boundaries."""
    cfg = dict(config)
    if cfg["mode"] not in {"smoke", "rehearsal", "full"}:
        raise ValueError("Mode must be smoke, rehearsal, or full")
    if cfg["mode"] == "full" and any(cfg[f"{split}_limit"] is not None for split in ("train", "validation", "test")):
        raise ValueError("Full mode must preserve the full official splits")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if resume is not None and not Path(resume).is_dir():
        raise ValueError("Resume must be a previous sentiment suite output directory containing all model checkpoints")
    if resume is not None and not any((Path(resume) / name / "checkpoints" / "last.pt").is_file() for name in MODEL_NAMES):
        raise ValueError("Resume directory contains no sentiment checkpoints")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume and any((output_dir / name / "checkpoints" / "last.pt").exists() for name in MODEL_NAMES):
        raise FileExistsError("Output already contains checkpoints. Use resume or a fresh output directory.")
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    _seed(cfg["seed"])
    records = _load_records(cfg)
    audit = audit_splits(records)
    vocabulary = build_vocabulary((row["text"] for row in records["train"]), cfg["vocabulary_size"], cfg["min_frequency"])
    # Paths may change between local/cloud/college machines; data bytes may not.
    fingerprint = _fingerprint(cfg, vocabulary, records)
    accepted_fingerprints = {fingerprint}
    legacy_fingerprints = {}
    # Validate every existing model before touching a same-directory run's files.
    if resume:
        for name in MODEL_NAMES:
            checkpoint_dir = Path(resume) / name / "checkpoints"
            if not (checkpoint_dir / "last.pt").exists():
                continue
            for filename in ("last.pt", "best.pt"):
                source = checkpoint_dir / filename
                if not source.is_file():
                    if filename == "best.pt" and not last_state["history"] and last_state.get("epoch_progress"):
                        # The first checkpoint can precede the first validation/best model.
                        continue
                    raise ValueError(f"Resume is incomplete: missing {name}/{filename}")
                state = torch.load(source, map_location="cpu", weights_only=False)
                if state["model_name"] != name or _resume_contract(state["config"]) != _resume_contract(cfg):
                    raise ValueError("Resume configuration/data fingerprint mismatch")
                expected_fingerprint = fingerprint
                if state.get("format_version", 1) == 1:
                    legacy_key = json.dumps(state["config"], sort_keys=True)
                    if legacy_key not in legacy_fingerprints:
                        legacy_fingerprints[legacy_key] = _fingerprint(state["config"], vocabulary, records, legacy=True)
                    expected_fingerprint = legacy_fingerprints[legacy_key]
                if state["fingerprint"] != expected_fingerprint or state["vocabulary"] != vocabulary:
                    raise ValueError("Resume configuration/data fingerprint mismatch")
                accepted_fingerprints.add(expected_fingerprint)
                if filename == "last.pt":
                    last_state = {"history": state["history"], "epoch_progress": state.get("epoch_progress")}
                del state
    datasets = {name: Reviews(rows, vocabulary, cfg["max_length"]) for name, rows in records.items()}
    _dump(output_dir / "data_audit.json", audit)
    _dump(output_dir / "vocabulary.json", vocabulary)
    _dump(output_dir / "preprocessing.json", {"description": "HTML/whitespace normalization, lowercase, contraction expansion, punctuation removal, customized stopword removal preserving negation; vocabulary fitted only on training rows.", "stopwords": sorted(STOPWORDS), "max_length": cfg["max_length"], "padding_id": PAD, "unknown_id": UNK})
    np.savez_compressed(output_dir / "split_ids.npz", **{name: np.asarray([row["id"] for row in rows]) for name, rows in records.items()})
    _dump(output_dir / "config.json", cfg)
    _dump(output_dir / "dataset_statistics.json", {name: {"count": len(data), "truncated_fraction": float(np.mean(np.asarray(data.lengths) > cfg["max_length"])), "mean_oov_rate": float(np.mean(data.oov_rates)), "mean_tokens_before_truncation": float(np.mean(data.lengths))} for name, data in datasets.items()})
    results, predictions, reference_y = {}, {}, None
    for name in MODEL_NAMES:
        metrics, y, probabilities = _train_one(name, cfg, datasets, vocabulary, output_dir, device, fingerprint, Path(resume) if resume else None, accepted_fingerprints)
        if reference_y is not None and not np.array_equal(reference_y, y):
            raise AssertionError("Test row order differs between models")
        reference_y = y
        results[name], predictions[name] = metrics, probabilities
    comparisons = {name: paired_mcnemar(reference_y, predictions["maxpool_mlp"], predictions[name]) for name in MODEL_NAMES[1:]}
    _dump(output_dir / "paired_mcnemar.json", comparisons)
    _dump(output_dir / "summary.json", {"mode": cfg["mode"], "synthetic": cfg["mode"] == "smoke", "models": results, "paired_mcnemar": comparisons, "fingerprint": fingerprint, "manual_error_review_complete": False})
    return {"mode": cfg["mode"], "synthetic": cfg["mode"] == "smoke", "models": results, "output_dir": str(output_dir), "manual_error_review_complete": False}
