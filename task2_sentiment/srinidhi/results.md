# Srinidhi - Yelp sentiment results

## Measured test results

| Model | Accuracy | Macro-F1 | Accuracy 95% bootstrap interval | ROC-AUC | MCC |
| --- | ---: | ---: | --- | ---: | ---: |
| maxpool_mlp | 93.20% | 0.9320 | 92.91%–93.46% | 0.9816 | 0.8640 |
| bilstm | 96.01% | 0.9601 | 95.81%–96.21% | 0.9933 | 0.9203 |
| dilated_cnn | 95.80% | 0.9580 | 95.59%–96.00% | 0.9922 | 0.9161 |

The selected BiLSTM has the highest measured accuracy and macro-F1 among these three models. Its accuracy exceeds the MLP by 2.81 percentage points and the CNN by 0.21 points. This is a finite, single-initialization comparison, not a claim of the best possible Yelp result. Exact paired McNemar p-values against the MLP are 1.80e-113 for BiLSTM and 1.78e-109 for CNN (two prespecified, unadjusted comparisons). No BiLSTM-versus-CNN significance test was prespecified.

Final inference ran on an Apple M3 CPU after the RTX 5090 pod stopped. Training hardware below remains the actual cloud hardware. The full notebook executed with zero errors and fresh checkpoint reloads; see `verification/part_b_notebook.json` at the repository root. An AI-assisted interpretation of 20 BiLSTM errors is in failure_analysis.md; student review remains pending.

All three models trained independently using 504,000 training reviews and 56,000 validation reviews; final evaluation covers all 38,000 official test reviews. Training hardware and configurations below come from each actual source run. CPU evidence sources are recorded in outputs/full/training_sources.json.

| Model | Actual training run | CPU | GPU | Configuration |
| --- | --- | --- | --- | --- |
| maxpool_mlp | `runs/part-b-full` | AMD EPYC 7543 32-Core Processor | NVIDIA GeForce RTX 5090 | [Exact settings](outputs/full/maxpool_mlp/training_config.json) |
| bilstm | `runs/part-b-tuning/bilstm_long_schedule_control` | AMD EPYC 7543 32-Core Processor | NVIDIA GeForce RTX 5090 | [Exact settings](outputs/full/bilstm/training_config.json) |
| dilated_cnn | `runs/part-b-full` | AMD EPYC 7543 32-Core Processor | NVIDIA GeForce RTX 5090 | [Exact settings](outputs/full/dilated_cnn/training_config.json) |

This is a derived evaluation suite assembled from separately preserved training runs. Selection rule: For each model family find the highest validation macro-F1. Among candidates within 0.001 inclusive of that maximum, prefer fewer parameters, then higher validation macro-F1. Break an exact remaining tie by source path and candidate ID. This is a practical preference, not a statistical significance test; test metrics never enter selection. All candidate records and selected checkpoint hashes are preserved in outputs/full/selection_manifest.json. Original run evaluations and logs remain unchanged.

The baseline max-pool MLP tests what unordered lexical evidence can achieve. The BiLSTM adds bidirectional sequence context; the residual dilated CNN learns local patterns over a wider receptive field. Each model learns its own embeddings from scratch. See each model's training_config.json for its actual widths, learning rates, dropout, batch size and stopping settings. The source history records the validation trajectory and selected epoch.

Confidence intervals use the bootstrap sample count in each evaluation metric file and IID test-row resampling; they do not measure variation across training seeds. PR-AUC uses trapezoidal integration; ECE uses 15 equal-width top-label confidence bins. McNemar tests compare paired predictions against the baseline, with two unadjusted p-values.

All required numeric results, including slice support, macro-F1 and error rate, are in metrics_report.csv. Peak memory is PyTorch's maximum allocated CUDA memory, not the entire device reservation. Training time sums the source run's measured training epochs; source provenance records total invocation elapsed time, without a separate setup/validation/export breakdown. Training throughput is processed training examples divided by those measured epoch times. These source runs overlapped other Part B jobs on one RTX 5090. Their measured training times and throughput describe that shared session and are not controlled, isolated architecture-speed comparisons. The initial benchmark is separate evidence in verification/sentiment_5090_benchmark.json.

Data length and class distributions, class balance, blanks and truncation/OOV statistics are in outputs/full/data_distributions.json and .png.

Duplicate reviews are audited in outputs/full/data_audit.json and retained under the frozen official-row protocol. Shared text across splits is a small potential source of score inflation; these results are not a duplicate-clean benchmark.

Original evaluation evidence: ../../runs/part-b-selected. The notebook reads the published copies in outputs/full; byte-identical training and evaluation logs are mapped to their original paths in outputs/full/raw_logs/manifest.json. Publication metadata records original and published hashes for the two metadata files; derived copies use portable repository-relative paths. Original training evidence is listed per model above. Checkpoint mapping: checkpoints/manifest.json. outputs/full/model_comparison.csv summarizes the three models. Each model's required_20_errors_for_review.csv contains five errors from each required category. Student review is recorded only by explicit student_reviewed values; newly generated interpretation fields remain blank and unreviewed. Existing annotations are retained for unchanged checkpoint predictions; annotated prior sets are archived under review_history. Teammate comparisons and the combined final report require the teammate's actual results.

## Reproduction commands

Run from the repository root after preparing the documented environment and frozen full-data cache. Each destination below must be new. These commands retrain the selected recipes; preserved source manifests record the original code and environment.

```bash
.venv/bin/python -m lab1.run --task sentiment --mode full --config task2_sentiment/srinidhi/outputs/full/reproduction/maxpool_mlp_full_config.json --device cuda --output runs/reproduce-part2-maxpool_mlp
.venv/bin/python scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/bilstm_candidate.json --output runs/reproduce-part2-bilstm --device cuda
```

To repeat final evaluation of the **preserved selected checkpoints**, use:

```bash
.venv/bin/python scripts/finalize_sentiment_selection.py --selection runs/part-b-selected/selection_manifest.json --output runs/reproduce-part2-final-evaluation
```

After retraining, create a new selection manifest from the new validation results, paths and checkpoint hashes before finalizing those new runs. The command above intentionally references the original frozen selection; it does not select the newly trained runs.

See results.md for metric definitions and limitations. Student error review and teammate comparisons are separate deliverables.
