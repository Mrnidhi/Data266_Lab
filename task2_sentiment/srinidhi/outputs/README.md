# Part 2 outputs

`full/` contains the October 1 desktop evaluation of three newly trained,
validation-selected checkpoints on all 38,000 official Yelp test reviews.
Training and final evaluation used Intel Core Ultra 9 285K / RTX 5090;
fresh saved-checkpoint checks also ran on CPU. Selected epochs are MLP 5,
BiLSTM 11 and CNN 6. Historical cloud/Mac publication is preserved under
`publication_history/b257a1079ef85973/` and is not the current result source.

- `model_comparison.csv` and the member-level `metrics_report.csv`: model and slice metrics, intervals and paired comparisons.
- `selection_manifest.json` and `training_sources.json`: three fixed desktop candidates, the frozen validation selection rule and exact per-model source runs.
- Model subfolders: predictions, training history, confusion matrix, ROC/PR and training plots, metrics and the required 20 errors for review.
- `data_distributions.json/png`: label counts, review lengths, truncation and OOV statistics.
- `raw_logs/manifest.json`: exact log copies and source hashes. `publication_metadata.json` documents portable path normalization in derived metadata copies.

Student error-review fields remain blank/false until the student reviews them.
Each model's separate `ai_error_review_draft.csv` contains twenty AI-assisted
interpretations keyed to example/checkpoint/source-text hashes. These and the
member's all-model failure analysis do not establish manual review. Human
packets remain byte-identical to those shown in the executed notebook.
Fresh source evidence is under
`reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/`.
The old cloud run remains historical evidence under
`reproducibility/raw_logs/srinidhi/runpod-20260918/part-b/`.
