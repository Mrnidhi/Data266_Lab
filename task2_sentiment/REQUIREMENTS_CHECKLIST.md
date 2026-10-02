# Part 2 — desktop plan and requirement mapping

Audit date: October 1, 2026. Scope: Srinidhi's Yelp Polarity sentiment work.
Requirements come from Task 2 on pages 7–8 of the provided DATA266 Lab1 Fall
2026 PDF, the individual/team instructions, and the supplied Canvas message.
The user authorized continuing Part 2 after Part 1. This document records the
desktop starting state, verified preparation and frozen reproduction plan.
Fresh full training started on October 1; its final metrics, publication,
notebook and package have not yet been established by this checklist.

## Starting audit and prepared desktop state

The previous cloud publication is present: three models' metrics, predictions,
plots, source/configurations, selection and training provenance, the combined
metric CSV, and the previously executed notebook. The notebook has seven
executed code cells with visible outputs and no saved error outputs. Those
outputs describe the earlier run, not a new execution on this desktop.

| Historical artifact | Initial audit result |
|---|---|
| `srinidhi/results.md` and `metrics_report.csv` | Earlier measured test accuracy: MLP 93.20%, BiLSTM 96.01%, CNN 95.80%; preserve these as historical results. |
| `srinidhi/checkpoints/manifest.json` | Six original best/last file hashes were recorded, but the corresponding weights were absent at the initial desktop audit. Fresh checkpoints now belong to the new run. |
| `srinidhi/data_processed/` and shared `data/` | Initially no full dataset cache was present. The full JSONL splits and encoded cache have now been rebuilt; the identity receipt verifies the same historical rows and ordering. |
| Earlier `dist/part-b-srinidhi-20260918-checkpoints.zip` | Not present on this desktop; the older README's backup description refers to the earlier machine. |
| Three `required_20_errors_for_review.csv` files | Exactly 20 distinct errors per model, five per required group. All `error_type`/`testable_fix` fields are blank and no row has `student_reviewed=True`. |
| `srinidhi/failure_analysis.md` | AI-assisted draft for 20 historical BiLSTM errors only; MLP/CNN interpretation and student review are incomplete. |

The full-data cache now contains 504,000 training, 56,000 validation and
38,000 test reviews. The resolved Hugging Face revision is
`bbf1c97a1f0cf005e5aded43839fd814654a1557`. All three split row contents and
ordering match the historical frozen dataset exactly. The desktop JSONL files
use CRLF; both actual byte hashes and normalized-LF hashes are recorded in
`verification/part2_desktop_data_identity.json`. The cache and new distribution
analysis receipts are `verification/part2_desktop_data_cache.json` and
`verification/part2_desktop_data_analysis/data_distributions.json`.

The historical notebook alone cannot finalize a desktop training/evaluation
package. The fresh run is now executing sequentially at
`reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/`.
Its `desktop_plan.json` freezes training source hashes, configuration and the
three `recipes/*.json` files before training. Preserve the historical
publication and its provenance until the new run is complete, validated and
independently identifiable.

The fresh MLP has completed six epochs and selected epoch 5 with validation
macro-F1 **0.9264224689043884**. This is a validation score, not a test result.
The remaining model runs, final selection and new test evaluation are still
separate completion checks; do not copy historical test scores into the new
run's report.

## Frozen desktop experiment

Train three separate models from scratch using the already selected historical
architectures and schedules. Each model's training function receives training
and validation tensors only. This repeats the selected recipes; it does not repeat
the earlier nine-candidate search or establish a new optimum.

| Role / model | Fixed architecture | Initial LR | Maximum epochs / patience |
|---|---|---:|---|
| Baseline `maxpool_mlp` | Own embedding 128, masked max pooling, dense 64/ReLU/dropout, binary logit | 0.0015 | 6 / 2 |
| Experimental `bilstm` | Own embedding 128, one bidirectional LSTM with 96 units per direction, masked max pooling, dense 64/dropout, logit | 0.001 | 12 / 4 |
| Experimental `dilated_cnn` | Own embedding 128, 128 channels, four residual blocks with two kernel-3 convolutions each and dilations 1/2/4/8, masked pooling, dense 64/dropout, logit | 0.0008 | 6 / 2 |

Common settings: seed 2342, training-only vocabulary capped at 40,000 words,
minimum frequency 2, maximum 384 tokens, batch size 128, AdamW weight decay
0.0001, gradient clipping 1.0, embedding/head dropout 0.3 and CNN-block dropout
0.2. Learning rate halves on a validation-loss plateau. Use `num_workers=0`
on the Windows desktop. Early stopping follows validation macro-F1 with the
existing minimum-improvement threshold of 0.0001; the saved best checkpoint
need not be the strict numerical maximum across every epoch.

The dataset/feature configuration remains unchanged. Official Yelp training
rows are stratified into 504,000 training and 56,000 validation reviews; all
38,000 official test reviews are held for final evaluation. The resolved
dataset revision, row IDs, labels, content hashes, split fingerprint,
preprocessing and training-only vocabulary are frozen before training. All three models
use the same prepared features and fit their own embedding weights.

The historical test results and error texts have already been inspected.
The repeated official test set is therefore not a newly sealed holdout.
Fresh checkpoint selection must use validation only. Freeze a new selection
manifest with the new checkpoint paths/hashes before calculating new test
predictions. Do not reuse the old selection manifest as evidence selecting
new weights, and do not choose further recipes from new test/error results.

The plan's shorthand “no new test access during training” means no new test
inference, predictions or metrics for training/checkpoint selection. The tuner
does read and verify the complete frozen split/cache fingerprints, including
test-row metadata and encoded arrays, before removing the test dataset from
the call to `_train_one`. It does not train on those test tensors. The frozen
plan remains unchanged; this explanation states the actual access boundary.

The Part 2 desktop environment receipt identifies an **Intel Core Ultra 9
285K** CPU, **NVIDIA GeForce RTX 5090**, Python 3.12.14, PyTorch 2.11.0+cu128
and CUDA 12.8. See `verification/part2_desktop_environment.json` and each
actual training provenance record. Do not substitute the older AMD EPYC cloud
CPU or Apple M3 inference CPU for this run. The Windows UTF-8 I/O and
host-memory/resume fixes preserve architectures, tokenization and training
hyperparameters. A native CUDA smoke and the relevant preflight/runner/
finalization test groups passed before full training. Test groups overlap;
their receipt counts are not a count of distinct tests.

## Execution and verification sequence

1. Record the current historical publication, then freeze the three new recipe
   files, source hashes, dataset revision and desktop hardware/environment.
2. Download and validate the full dataset; regenerate frozen JSONL and encoded
   caches. Verify row counts, ID disjointness, labels, text hashes and
   training-only vocabulary. Recompute distributions, malformed/empty counts,
   duplicate audit, truncation and OOV statistics for this prepared dataset.
3. Run native Windows smoke and relevant sentiment/preprocessing/resume tests.
   From the repository root after the documented pinned `.venv` setup:

   ```powershell
   .venv\Scripts\python.exe -m lab1.run --task sentiment --mode smoke --device cpu
   ```

4. Run each frozen recipe through `scripts/tune_sentiment.py` in its own fresh
   output directory on CUDA. Save unedited logs, actual environment, histories
   and both best/last checkpoints. Do not pass a test dataset to training.
5. Freeze the new selection manifest after all three validation-only runs
   complete. Check each selected file's hash, configuration, vocabulary,
   progress and validation record against its source run.
6. Evaluate all selected models on the identical 38,000 test IDs. Generate
   every metric below, paired comparisons, slices, curves and predictions.
   Verify arithmetic, finite/undefined-value handling and checkpoint mapping.
7. Generate each model's 20-case manual-review packet from those new
   predictions. Keep historical packets and annotations separately; a changed
   checkpoint/prediction does not inherit a completed human review.
8. Publish the new results under an explicit desktop run identity and execute
   its notebook with visible outputs. Reload each actual selected checkpoint
   for fresh inference, then verify weights/data/output hashes and the Part 2
   ZIP inventory before moving or pushing the publication.

The launched end-to-end command is:

```powershell
.venv\Scripts\python.exe scripts/run_sentiment_desktop.py --output reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full --stage all --device cuda
```

It verifies or freezes the desktop plan, trains the three families sequentially,
freezes `selection_manifest.json`, then evaluates into `selected/`. Do not
launch a second copy against an actively running output directory. A later
restart checks the frozen source/config/recipes, keeps completed invocations
and resumes unfinished ones from their last checkpoints. Changed source or
recipes are rejected rather than silently accepted.

Exact full-run recipe paths and commands must match the new frozen plan and
actual receipts. The ordinary `--mode full` reference profile has a six-epoch
maximum for all families; it does not reproduce the selected twelve-epoch
BiLSTM schedule by itself. The existing nine-candidate selector also expects
the original search layout; a fresh three-run selection must describe only
the actual new candidates.

## PDF requirement mapping

| Requirement | Existing implementation/evidence | Fresh desktop completion needed |
|---|---|---|
| Review-length and class-balance analysis | New `verification/part2_desktop_data_analysis/` JSON and PNG | Preparation recomputed; include them in the final publication. |
| Missing/malformed text or labels | Source rejects malformed rows and records empty counts; empty processed text maps to UNK | Verify actual new counts and report the handling policy. |
| Lowercasing, punctuation/special-character filtering and stopwords | Casefolding, HTML/contraction cleanup, regex word tokens and frozen customized stopwords; negation/contrast words retained | Preserve the frozen preprocessing and demonstrate it in the new notebook. |
| Stemming or lemmatization, if applicable | Not applied by the current tokenizer; the brief makes this conditional | Explain the decision; do not claim an unimplemented transformation. |
| Tokenization and embeddings learned from scratch | Training-only word dictionary, separate random embeddings for each model | Verify new vocabulary isolation and absence of pretrained weights. |
| Three own distinct models: baseline and two experiments | Max-pool MLP, BiLSTM, residual dilated CNN | Train all three new recipes; obtain teammate models for the required cross-member distinction. |
| Architecture/embedding justification | Existing member README and historical results | Student verifies the choices and new evidence; AI text is not proof of independent understanding. |
| Exact CPU/GPU for each training run | Intel Core Ultra 9 285K / RTX 5090 in the desktop environment receipt | Environment captured; verify each completed model's source mapping. |
| All evaluation metrics per model | Existing evaluator and metric CSV cover the full list below | Recompute on the new selected weights and common test IDs. |
| Twenty manually reviewed errors per model | Correctly grouped historical CSV packets; human fields blank | Student reviews 20 new errors for each model, types each and proposes a testable fix. |
| Within-member comparison, limitations and future work | Historical three-model comparison is available | Compare actual new outcomes and document uncertainty/protocol limits. |
| Team comparisons and joint analysis | Teammate folders are contribution scaffolds | Obtain independent teammate results; complete combined tables and synthesis. |
| Executed notebook, weights and required outputs | Historical executed notebook exists; actual weights/data missing here | New executed notebook, verified trained files and complete Part 2 folder/ZIP. |
| Raw logs, environment/config and checkpoint-result mapping | Historical published raw-log/provenance copies exist | Preserve new raw logs unchanged; record portable paths and file hashes. |
| GitHub publication | Code/results repository exists; app repository access remains unresolved | Verify the actual new remote commit and its required files before claiming a push. |

## Every required metric, for each model

| Metric family | Required values / definition |
|---|---|
| Classification | Accuracy; precision, recall and F1 for macro, micro and weighted averages; confusion matrix |
| Ranking | ROC-AUC and trapezoidal PR-AUC from positive-class probabilities |
| Correlation and calibration | MCC, Brier score and ECE; current ECE uses 15 equal-width top-label-confidence bins |
| Uncertainty | 95% bootstrap intervals for accuracy, macro-F1 and MCC; full profile uses 1,000 paired-index IID resamples |
| Paired tests | Exact two-sided McNemar comparisons of baseline MLP against BiLSTM and CNN on identical test rows; report the two unadjusted tests explicitly |
| Robustness | Slice support, macro-F1 and error rate for the predefined review-length, negation and OOV slices |
| Cost and hardware | Parameter count, actual training time, examples/second and peak memory, with measurement scope and CPU/GPU provenance |

Bootstrap intervals describe test-row sampling variation, not variation across
training seeds. Single-class slices may have undefined ROC/PR-AUC; preserve
null values with a reason. GPU allocator peak and host peak RSS are different
measurements. Times from the older overlapping cloud jobs are not measurements
of isolated speed on this desktop.

## Manual review: 60 model-specific entries

The required review is **20 errors for each of the three models**. Each model
needs five confident false positives, five confident false negatives, five
near-threshold errors and five slice-specific failures. The existing pipeline
uses a prespecified long-review slice for the last group and avoids duplicate
IDs within each model's packet. The same review may legitimately be an error
for multiple models; 60 entries do not necessarily mean 60 unique review texts.

Inspect the full review, ground truth, prediction, probability, preprocessing
and truncation before entering `error_type` and one `testable_fix`. Record
`student_reviewed=True` only after the student actually reviews that case.
AI draft columns and automatically selected examples do not complete manual
review. When a required group lacks five actual errors, report the shortfall;
never invent errors or replace them with correct predictions.

The historical BiLSTM has 20 AI-assisted interpretations; neither those drafts
nor historical model accuracy establish completed student review. Test-error
fixes are future hypotheses to evaluate with training/validation data and an
appropriate new evaluation, not authorization to tune against this test set.

## Remaining submission and ownership obligations

The brief requires each student's own core architecture decisions/analysis,
independent models from every member, and individual viva understanding.
Assisted implementation and automated training do not establish those facts;
see the root `AI_USE.md`. The student must defend the model choices, metric
calculation and error-review observations honestly.

The final Canvas submission is one ZIP containing separate Part 1, Part 2 and
Part 3 folders plus one combined `Report.pdf` with the GitHub link. Every part
must have its executed notebook, actual weights and required outputs. The
combined report needs all members' comparisons, ownership, traceable evidence,
joint limitations and references. A technically verified Part 2 package does
not complete manual student review, teammate work, the report or submission.
