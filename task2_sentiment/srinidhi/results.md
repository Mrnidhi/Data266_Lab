# Srinidhi - Yelp sentiment results

All three models trained independently using 504,000 training reviews and 56,000 validation reviews; final evaluation covers all 38,000 official test reviews. Training hardware and configurations below come from each actual source run. CPU evidence sources are recorded in outputs/full/training_sources.json.

These are the fresh October 1, 2026 desktop results. Earlier cloud/Mac
publication is preserved in `outputs/publication_history/b257a1079ef85973/`.
The fixed three-recipe reproduction does not repeat the historical search;
the official test set's historical scores and error texts were already observed.

| Model | Selected epoch | Validation macro-F1 | Test accuracy (95% IID bootstrap CI) | Test macro-F1 | MCC |
|---|---:|---:|---:|---:|---:|
| MLP | 5 | 0.926422 | 93.1921% (92.9211–93.4500%) | 0.931917 | 0.863953 |
| BiLSTM | 11 | 0.958857 | 96.1211% (95.9131–96.3079%) | 0.961210 | 0.922425 |
| Dilated CNN | 6 | 0.953446 | 95.6263% (95.4236–95.8184%) | 0.956262 | 0.912564 |

| Model | ROC-AUC | Trapezoidal PR-AUC | Brier | ECE (15 bins) | Parameters | Epoch training seconds | Training examples/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.981637 | 0.982372 | 0.052846 | 0.027214 | 5,128,321 | 86.9275 | 34,787.6 |
| BiLSTM | 0.993401 | 0.993509 | 0.029803 | 0.014531 | 5,305,985 | 2,911.4808 | 2,077.3 |
| Dilated CNN | 0.991775 | 0.991868 | 0.033397 | 0.012759 | 5,539,073 | 240.7418 | 12,561.2 |

BiLSTM improves test accuracy over MLP by 2.93 percentage points and over CNN
by 0.49 in this single-seed replication. It also has the best Brier score;
CNN has the lowest ECE. The exact paired baseline McNemar comparisons give
p-values 1.30349e-124 for BiLSTM and 1.54979e-92 for CNN, with two unadjusted
tests. These differences do not establish seed stability or new-domain quality.
Sequence modeling takes more measured training time; all resource figures
need the measurement limits below. Complete averaging modes, confusion
matrices, intervals, slice support/F1/error rate and memory fields remain in
the machine-readable metric report and per-model files.

| Model | Actual training run | CPU | GPU | Configuration |
| --- | --- | --- | --- | --- |
| maxpool_mlp | `reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/training/maxpool_mlp` | Intel(R) Core(TM) Ultra 9 285K | NVIDIA GeForce RTX 5090 | [Exact settings](outputs/full/maxpool_mlp/training_config.json) |
| bilstm | `reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/training/bilstm` | Intel(R) Core(TM) Ultra 9 285K | NVIDIA GeForce RTX 5090 | [Exact settings](outputs/full/bilstm/training_config.json) |
| dilated_cnn | `reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/training/dilated_cnn` | Intel(R) Core(TM) Ultra 9 285K | NVIDIA GeForce RTX 5090 | [Exact settings](outputs/full/dilated_cnn/training_config.json) |

This is a derived evaluation suite assembled from separately preserved training runs. Selection rule: Architectures and schedules were fixed before desktop training from the historical Part 2 selection: original MLP and CNN, original BiLSTM with 12 epochs/patience 4. For each family use its best validation macro-F1 checkpoint, with the configured minimum improvement and early stopping. No fresh test predictions or test metrics are used to select checkpoints. All candidate records and selected checkpoint hashes are preserved in outputs/full/selection_manifest.json. Original run evaluations and logs remain unchanged.

The baseline max-pool MLP tests what unordered lexical evidence can achieve. The BiLSTM adds bidirectional sequence context; the residual dilated CNN learns local patterns over a wider receptive field. Each model learns its own embeddings from scratch. See each model's training_config.json for its actual widths, learning rates, dropout, batch size and stopping settings. The source history records the validation trajectory and selected epoch.

Confidence intervals use the bootstrap sample count in each evaluation metric file and IID test-row resampling; they do not measure variation across training seeds. PR-AUC uses trapezoidal integration; ECE uses 15 equal-width top-label confidence bins. McNemar tests compare paired predictions against the baseline, with two unadjusted p-values.

All required numeric results, including slice support, macro-F1 and error rate, are in metrics_report.csv. GPU peak memory is PyTorch's maximum allocated CUDA memory, not the entire device reservation. Host RSS is separately recorded as the process-lifetime peak including data preparation; evaluation-process host peaks are separate from training-process peaks. Training time sums the source run's measured training epochs; source provenance records total invocation elapsed time, without a separate setup/validation/export breakdown. Training throughput is processed training examples divided by those measured epoch times. Training times and throughput describe each recorded source invocation. Use its timestamps and hardware provenance to establish whether workloads overlapped before comparing architecture speed. These measurements do not constitute a controlled isolated-speed benchmark.

Data length and class distributions, class balance, blanks and truncation/OOV statistics are in outputs/full/data_distributions.json and .png.

Duplicate reviews are audited in outputs/full/data_audit.json and retained under the frozen official-row protocol. Shared text across splits is a small potential source of score inflation; these results are not a duplicate-clean benchmark.

Original evaluation evidence, relative to the repository root:
`reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/selected`.
The notebook reads published copies in `outputs/full`; byte-identical histories
and provenance map to original paths in `outputs/full/raw_logs/manifest.json`.
Publication metadata records original/published hashes for derived portable
metadata. Original training sources are listed above; checkpoint mapping is
`checkpoints/manifest.json`. Each model's `required_20_errors_for_review.csv`
contains five actual errors per required category. Separate
`ai_error_review_draft.csv` files and [failure analysis](failure_analysis.md)
cover all sixty entries with AI-assisted types, exact evidence, explanations
and one future testable fix. Original human packets and the executed notebook
remain byte-identical; human interpretation fields are blank and all
`student_reviewed` values are false. `verification/part2_ai_error_drafts.json`
records linkage checks. Existing historical annotations stay bound to their
own checkpoint/predictions. Student manual review, teammate comparisons and
combined report require actual student/teammate work.

## Reproduction commands

Run from the repository root using the activated Python environment and frozen full-data cache. Each destination below must be new. These commands retrain the selected recipes; preserved source manifests record the original code and environment.

```bash
python scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/maxpool_mlp_candidate.json --output runs/reproduce-part2-maxpool_mlp --device cuda
python scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/bilstm_candidate.json --output runs/reproduce-part2-bilstm --device cuda
python scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/dilated_cnn_candidate.json --output runs/reproduce-part2-dilated_cnn --device cuda
```

To repeat final evaluation of the **preserved selected checkpoints**, use:

```bash
python scripts/finalize_sentiment_selection.py --selection reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/selected/selection_manifest.json --output runs/reproduce-part2-final-evaluation
```

After retraining, create a new selection manifest from the new validation results, paths and checkpoint hashes before finalizing those new runs. The command above intentionally references the original frozen selection; it does not select the newly trained runs.

See [metric report](metrics_report.csv) for complete numerical results and
[failure analysis](failure_analysis.md) for explicitly assisted hypotheses.
The [finalization guide](../../PART2_FINALIZATION.md) records native Windows
commands and package verification. Student review and teammate comparison
remain separate deliverables.
