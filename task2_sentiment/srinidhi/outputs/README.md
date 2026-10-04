# Part 2 outputs

`full/` contains the current evaluation of the validation-selected wider MLP, retained BiLSTM and original CNN on all 38,000 official test reviews. Selected epochs are 5, 11 and 6; source histories completed 9, 12 and 6 epochs respectively.

- `summary.json`, `model_comparison.csv` and member-level `metrics_report.csv` contain measured results.
- `selection_manifest.json` records all seven candidates, the frozen 0.001 validation tolerance and selected hashes. The original CNN remains selected because the wider CNN's gain was smaller than that tolerance.
- `training_sources.json` and per-model `training_config.json` identify actual settings, CPU/GPU and source runs.
- Model subfolders contain predictions, histories, metrics, curves, training plots and twenty errors each. Required metrics include Brier/ECE, bootstrap intervals, McNemar comparisons, slices, training time, examples/second and memory.
- `data_distributions.json/png` and `data_audit.json` document balance, lengths, truncation, OOV and duplicates.
- `raw_logs/manifest.json` maps portable logs to sources; `publication_metadata.json` records path-normalized metadata and original/published hashes.

The selected suite is `reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected/`. Compact recipes and completed candidate evidence are retained in `quality_search/desktop-quality-20261001/`.

Each `ai_error_review_draft.csv` provides AI-assisted hypotheses and future testable fixes, linked to current packet, checkpoint and text hashes. Human `error_type` and `testable_fix` fields remain blank and all 60 `student_reviewed` flags are false. These drafts support student review; they do not complete it.

Earlier publications remain local and in Git history. The current ZIP retains selected results and compact candidate evidence. Metric, notebook, saved-model and extracted-package checks passed for the selected suite. See the [finalization guide](../README.md).
