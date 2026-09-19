# Part 2 outputs

`full/` contains the final evaluation of the three validation-selected checkpoints on all 38,000 official Yelp test reviews. Training used the RunPod RTX 5090; final inference used the local Apple M3 CPU after the pod stopped.

- `model_comparison.csv` and the member-level `metrics_report.csv`: model and slice metrics, intervals and paired comparisons.
- `selection_manifest.json` and `training_sources.json`: all nine candidates, the frozen selection rule and exact per-model source runs.
- Model subfolders: predictions, training history, confusion matrix, ROC/PR and training plots, metrics and the required 20 errors for review.
- `data_distributions.json/png`: label counts, review lengths, truncation and OOV statistics.
- `raw_logs/manifest.json`: exact log copies and source hashes. `publication_metadata.json` documents portable path normalization in derived metadata copies.

Student error-review fields remain false until the student reviews them. AI draft interpretations are identified separately. The original cloud run evidence is preserved under `reproducibility/raw_logs/srinidhi/runpod-20260918/part-b/`.
