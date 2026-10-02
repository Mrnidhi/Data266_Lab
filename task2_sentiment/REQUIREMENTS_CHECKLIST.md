# Part 2 — Lab 1 requirement checklist

Scope: Srinidhi's Task 2 Yelp Polarity work, following pages 7–8 of the supplied DATA266 Lab1 Fall 2026 PDF and individual/team submission instructions. This checklist distinguishes published evidence from student and team work still requiring completion.

The current suite is `reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected/`. It replaces the MLP with a wider model and retains the desktop BiLSTM and CNN.

| Model | Completed / selected epoch | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| MLP | 9 / 5 | 0.928285 | 93.310526% | 0.933105 |
| BiLSTM | 12 / 11 | 0.958857 | 96.121053% | 0.961210 |
| CNN | 6 / 6 | 0.953446 | 95.626316% | 0.956262 |

The frozen rule considers seven candidates and prefers fewer parameters within 0.001 of each family's highest validation macro-F1. The wider CNN improves by only 0.000733, so the original CNN is retained. New test predictions did not select models. Earlier test results/texts were observed; this is not a new unseen holdout. Historical publications remain under `srinidhi/outputs/publication_history/`.

## Required evidence

| PDF requirement | Current evidence | Student or finalization action |
|---|---|---|
| Lengths, class balance, missing/malformed data | `outputs/full/data_distributions.json`, `.png`, `data_audit.json`; balanced 504,000/56,000/38,000 splits; no blank raw rows | Explain distributions and mapping 22 training/1 validation empty processed texts to UNK. |
| Text preprocessing and conditional stemming/lemmatization | Casefolding, HTML/contraction handling, punctuation and customized stopword filtering; negation retained | Explain omitted stemming and training-only vocabulary. |
| Baseline plus two distinct experiments; own learned embeddings | Independently initialized max-pool MLP, BiLSTM and residual dilated CNN | Explain choices and differences; AI assistance does not establish independent student design. |
| Architectures and settings | Per-model `training_config.json`, `training_sources.json`, histories and checkpoint hashes | MLP uses embedding 256/head 128/dropout 0.5; BiLSTM/CNN retain embedding 128. |
| Exact training hardware and cost | Source provenance: Intel Core Ultra 9 285K / RTX 5090; parameters, time, throughput, memory | Explain measurement scope and workload overlap before comparing speed. |
| Complete evaluation on matching test examples | `metrics_report.csv`, per-model metrics/predictions/curves, summary and paired tests | Fresh finalization has verified the current source/data/checkpoint mapping and metrics. |
| Twenty manually reviewed errors per model | Three 20-case packets and separate AI drafts with exact quotes | All 60 human flags remain false; student review is pending. |
| Within-member comparison and future fixes | `results.md`, AI-assisted `failure_analysis.md` | Verify conclusions in personal words; fixes are future training/validation studies. |
| Executed notebook, weights and outputs | Current publication and six local best/last files | Fresh notebook and saved-model checks passed; rebuild the ZIP and verify extracted execution. |
| Reproducibility and ownership | Frozen recipes/selection, source histories, hashes and AI disclosure | Preserve original evidence; explain implementation and analysis in the viva. |
| Independent teammates and joint report | Member contribution folders | Obtain actual teammate runs, compare them and complete the joint report. |

Evidence paths in the table are relative to `task2_sentiment/srinidhi/` unless stated otherwise.

## Every required metric, per model

| Family | Values and definitions |
|---|---|
| Classification | Accuracy; macro/micro/weighted precision, recall and F1; confusion matrix |
| Ranking | ROC-AUC and trapezoidal PR-AUC from positive-class probabilities |
| Correlation and calibration | MCC, Brier score and 15-bin top-label-confidence ECE |
| Uncertainty | 95% accuracy, macro-F1 and MCC intervals from 1,000 IID test-row bootstrap resamples |
| Paired comparison | Exact two-sided McNemar tests: MLP versus BiLSTM and MLP versus CNN; two unadjusted p-values |
| Slices | Support, macro-F1 and error rate for predefined length, negation and OOV groups |
| Cost | Parameter count, measured training time, examples/second and peak memory with actual CPU/GPU |

Bootstrap intervals describe test-row variation, not training-seed variation. Preserve undefined single-class values as null. GPU allocator peak and host peak RSS measure different things. Seven text hashes are shared between training and test, so this is not a duplicate-clean benchmark.

## Student review and submission

Each model's `required_20_errors_for_review.csv` contains five confident false positives, five confident false negatives, five near-threshold errors and five long-review errors. Inspect the full text, label, probability and retained input; provide an error type, grounded explanation and one testable fix. Set `student_reviewed=True` only after doing that review. The same text can appear for more than one model.

AI drafts retain `AI_ASSISTED_NOT_STUDENT_REVIEW`; they do not satisfy manual review or prove a failure mechanism. Their linkage receipt is `verification/part2_ai_error_drafts.json`. Proposed changes require training/validation studies rather than selection from these test errors.

Follow [PART2_FINALIZATION.md](../PART2_FINALIZATION.md) for commands and receipts. Finalization must package actual selected weights, notebook, data and outputs, then verify the ZIP and extracted execution. The wider MLP's exact LFS path is `task2_sentiment/srinidhi/checkpoints/maxpool_mlp/best.pt`; clone users need `git lfs pull`. A normal final push to `main` is authorized, preserving history.

Canvas requires one combined ZIP with Part 1, Part 2 and Part 3 folders and one combined `Report.pdf` containing the GitHub link. A Part 2-only ZIP does not complete teammate work, student review, the report or submission. See [AI_USE.md](../AI_USE.md) for assistance disclosure.
