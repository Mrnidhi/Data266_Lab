# Srinidhi - Yelp sentiment results

All three models trained independently using 504,000 training reviews and 56,000 validation reviews; final evaluation covers all 38,000 official test reviews. Training hardware and configurations below come from each actual source run. CPU evidence sources are recorded in outputs/full/training_sources.json.

| Model | Test accuracy | Test macro-F1 | Parameters |
|---|---:|---:|---:|
| maxpool_mlp | 93.310526% | 0.933105 | 10,273,025 |
| bilstm | 96.121053% | 0.961210 | 5,305,985 |
| dilated_cnn | 95.626316% | 0.956262 | 5,539,073 |

| Model | Actual training run | CPU | GPU | Configuration |
| --- | --- | --- | --- | --- |
| maxpool_mlp | `reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/training/mlp_wide_regularized` | Intel(R) Core(TM) Ultra 9 285K | NVIDIA GeForce RTX 5090 | [Exact settings](outputs/full/maxpool_mlp/training_config.json) |
| bilstm | `reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/training/bilstm` | Intel(R) Core(TM) Ultra 9 285K | NVIDIA GeForce RTX 5090 | [Exact settings](outputs/full/bilstm/training_config.json) |
| dilated_cnn | `reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/training/dilated_cnn` | Intel(R) Core(TM) Ultra 9 285K | NVIDIA GeForce RTX 5090 | [Exact settings](outputs/full/dilated_cnn/training_config.json) |

This is a derived evaluation suite assembled from separately preserved training runs. Selection rule: For each family, consider its existing desktop control and every completed new candidate. Find the highest selected-checkpoint validation macro-F1. Among candidates within 0.001 inclusive of that maximum, prefer fewer parameters, then higher validation macro-F1, then source path and candidate ID. A wider candidate must therefore exceed a smaller control by more than 0.001 to displace it. Checkpoints follow configured early_stopping_min_delta=0.0001. No test score or test prediction enters selection. This practical tolerance does not establish statistical significance. All candidate records and selected checkpoint hashes are preserved in outputs/full/selection_manifest.json. Original run evaluations and logs remain unchanged.

The baseline max-pool MLP tests what unordered lexical evidence can achieve. The BiLSTM adds bidirectional sequence context; the residual dilated CNN learns local patterns over a wider receptive field. Each model learns its own embeddings from scratch. See each model's training_config.json for its actual widths, learning rates, dropout, batch size and stopping settings. The source history records the validation trajectory and selected epoch.

Confidence intervals use the bootstrap sample count in each evaluation metric file and IID test-row resampling; they do not measure variation across training seeds. PR-AUC uses trapezoidal integration; ECE uses 15 equal-width top-label confidence bins. McNemar tests compare paired predictions against the baseline, with two unadjusted p-values.

All required numeric results, including slice support, macro-F1 and error rate, are in metrics_report.csv. Peak memory is PyTorch's maximum allocated CUDA memory, not the entire device reservation. Training time sums the source run's measured training epochs; source provenance records total invocation elapsed time, without a separate setup/validation/export breakdown. Training throughput is processed training examples divided by those measured epoch times. Training times and throughput describe each recorded source invocation. Use its timestamps and hardware provenance to establish whether workloads overlapped before comparing architecture speed. These measurements do not constitute a controlled isolated-speed benchmark.

Data length and class distributions, class balance, blanks and truncation/OOV statistics are in outputs/full/data_distributions.json and .png.

Duplicate reviews are audited in outputs/full/data_audit.json and retained under the frozen official-row protocol. Shared text across splits is a small potential source of score inflation; these results are not a duplicate-clean benchmark.

Original evaluation evidence: ../../reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected. The notebook reads the published copies in outputs/full; byte-identical training and evaluation logs are mapped to their original paths in outputs/full/raw_logs/manifest.json. Publication metadata records original and published hashes for the two metadata files; derived copies use portable repository-relative paths. Original training evidence is listed per model above. Checkpoint mapping: checkpoints/manifest.json. outputs/full/model_comparison.csv summarizes the three models. Each model's required_20_errors_for_review.csv contains five errors from each required category. Student review is recorded only by explicit student_reviewed values; newly generated interpretation fields remain blank and unreviewed. Existing annotations are retained for unchanged checkpoint predictions; annotated prior sets are archived under review_history. Teammate comparisons and the combined final report require the teammate's actual results.

## Overfitting check

The MLP shows late overfitting: from its selected epoch 5 to epoch 9, training loss fell from 0.1698 to 0.1098, while validation loss rose from 0.1860 to 0.2140 and validation macro-F1 fell from 0.9283 to 0.9258. Early stopping ended training at epoch 9; the saved model is epoch 5. BiLSTM validation loss rose from 0.1172 at epoch 11 to 0.1222 at epoch 12, with slightly lower macro-F1, so epoch 11 is retained. CNN validation loss was lowest at epoch 3, but its validation macro-F1 continued improving through the selected epoch 6. These checkpoints follow the declared macro-F1 selection rule. Training dropout affects the loss comparison, and these curves do not establish an absence of overfitting.

## Reproduction commands

Run from the repository root using the activated Python environment and frozen full-data cache. Each destination below must be new. These commands retrain the selected recipes; preserved source manifests record the original code and environment.

```bash
python scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/maxpool_mlp_candidate.json --output runs/reproduce-part2-maxpool_mlp --device cuda
python scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/bilstm_candidate.json --output runs/reproduce-part2-bilstm --device cuda
python scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/dilated_cnn_candidate.json --output runs/reproduce-part2-dilated_cnn --device cuda
```

To repeat final evaluation of the **preserved selected checkpoints**, use:

```bash
python scripts/finalize_sentiment_selection.py --selection reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected/selection_manifest.json --output runs/reproduce-part2-final-evaluation
```

After retraining, create a new selection manifest from the new validation results, paths and checkpoint hashes before finalizing those new runs. The command above intentionally references the original frozen selection; it does not select the newly trained runs.

See metrics_report.csv and each model's metrics.json for full measures and metric definitions. Student error review and teammate comparisons are separate deliverables.
