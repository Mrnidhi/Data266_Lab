"""Describe the verified Yelp splits without changing training data."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1 import sentiment as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--encoded-cache", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "task2_sentiment/srinidhi/config.json").read_text())["full"]
    cfg.update(data_dir=args.data_dir, encoded_cache=args.encoded_cache)
    records = s._load_records(cfg)
    vocabulary, datasets = s._load_features(cfg, records)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"vocabulary_size": len(vocabulary), "length_definition": "Tokens after the documented preprocessing, before truncation", "splits": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, (name, data) in enumerate(datasets.items()):
        lengths = np.asarray(data.lengths)
        labels = np.asarray([r["label"] for r in data.records])
        counts = np.bincount(labels, minlength=2)
        report["splits"][name] = {
            "count": len(data), "negative": int(counts[0]), "positive": int(counts[1]),
            "positive_fraction": float(labels.mean()), "blank_raw_reviews": sum(not r["text"].strip() for r in data.records),
            "empty_after_preprocessing": int((lengths == 0).sum()), "null_or_malformed_rows": 0,
            "validation_policy": "Reject missing/malformed text, labels or IDs; no rows silently removed",
            "length_quantiles": {str(q): float(np.quantile(lengths, q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)},
            "truncated_fraction": float((lengths > cfg["max_length"]).mean()),
            "mean_oov_rate": float(np.mean(data.oov_rates))}
        axes[0].hist(lengths, bins=np.arange(0, 2001, 50), density=True, histtype="step", label=name)
        axes[1].bar(np.arange(2) + (index - 1) * .25, counts, width=.25, label=name)
    axes[0].axvline(cfg["max_length"], color="black", linestyle="--", label="384-token cap")
    axes[0].set(xlabel="Processed token count (plot limited to 2,000)", ylabel="Density", title="Review length distribution")
    axes[1].set(xticks=[0, 1], xticklabels=["Negative", "Positive"], ylabel="Reviews", title="Class distribution")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "data_distributions.png", dpi=160)
    plt.close(fig)
    (args.output / "data_distributions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
