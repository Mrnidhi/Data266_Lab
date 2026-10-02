# Sentiment classification — Revanth0211

This folder contains my completed DATA 266 Lab 1 Part 2 Yelp Polarity experiment. I trained a learned mean-embedding baseline, a multi-kernel CNN, and a bidirectional GRU from scratch on the same train, validation, and test split.

The multi-kernel CNN was the best model, reaching 0.93775 test accuracy and 0.93775 macro-F1. Its improvement over the baseline was statistically distinguishable by the paired McNemar test (p = 1.25e-08).

## Contents

- `src/Part2.ipynb`: canonical executed notebook with visible outputs
- `config.json`: shared data and model settings
- `checkpoints/`: selected checkpoint for each of the three models
- `metrics_report.csv`: complete model comparison with uncertainty, calibration, and confusion counts
- `outputs/`: predictions, plots, slice metrics, vocabulary, and 20-row error review
- `results.md`: measured comparison and interpretation
- `failure_analysis.md`: summary of the manually reviewed errors
- `RUN_LOG.txt`: preserved run log

The raw dataset is intentionally not committed. The notebook downloads `fancyzhx/yelp_polarity` through Hugging Face Datasets.

Run the notebook from `task2_sentiment/revanth0211/src/` so its relative paths resolve to this member folder.
