# Part 2 — Yelp sentiment, Srinidhi

The October 1, 2026 desktop run freshly trained three separate classifiers
with embeddings learned from scratch. An Intel Core Ultra 9 285K CPU and
NVIDIA GeForce RTX 5090 ran fixed recipes sequentially using Python 3.12.14,
PyTorch 2.11.0+cu128 and CUDA 12.8. Frozen plan and untouched evidence:
`reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/`.

| Model | Completed / selected epoch | Validation macro-F1 | Test accuracy | Test macro-F1 | Parameters |
|---|---:|---:|---:|---:|---:|
| MLP | 6 / 5 | 0.926422 | 93.1921% | 0.931917 | 5,128,321 |
| BiLSTM | 12 / 11 | 0.958857 | 96.1211% | 0.961210 | 5,305,985 |
| Dilated CNN | 6 / 6 | 0.953446 | 95.6263% | 0.956262 | 5,539,073 |

All use the same 504,000 train, 56,000 validation and 38,000 official test
reviews. New checkpoints were frozen from validation before new test inference.
The repeated test is not a newly sealed holdout: earlier cloud scores/error
texts were already observed. Architectures/schedules were fixed beforehand
from historical selection; this is a three-recipe replication, not a fresh
hyperparameter search. Historical nine-candidate results/annotations are
preserved in `outputs/publication_history/b257a1079ef85973/` and described in
[research and search](RESEARCH_AND_SEARCH.md).

## Architecture and preprocessing

| Role | Architecture | Initial LR / maximum epochs / patience |
|---|---|---|
| Baseline MLP | Embedding 128 → masked max pool → dense 64/ReLU/dropout → binary logit | 0.0015 / 6 / 2 |
| Experiment BiLSTM | Embedding 128 → bidirectional LSTM, 96 units per direction → masked max pool → dense 64/dropout → logit | 0.001 / 12 / 4 |
| Experiment CNN | Embedding 128 → 128 channels → four residual blocks with two kernel-3 convolutions each, dilations 1/2/4/8 → masked pool → dense 64/dropout → logit | 0.0008 / 6 / 2 |

MLP tests unordered lexical evidence; BiLSTM adds sequence context; CNN learns
local composition over an expanded receptive field. Embedding width 128 is a
compact trainable representation held constant, not a pretrained language
representation. Each family independently initializes its own model/embedding.
BiLSTM packs real lengths; CNN masks padding after each convolution; all mask
pooling. These preserve padding invariance.

Common settings: seed 2342, training-only vocabulary cap 40,000, minimum
frequency two, first 384 tokens, batch 128, AdamW weight decay 0.0001, clipping
1.0, embedding/head dropout 0.3 and CNN-block dropout 0.2. Learning rate halves
on validation-loss plateau. Best checkpoints require validation macro-F1
improvement greater than 0.0001; the selected epoch need not be the strict
numerical maximum. Windows uses zero loader workers.

The deterministic tokenizer casefolds, expands contractions, removes HTML,
normalizes URLs and uses word tokens with customized stopwords, retaining
negation/contrast. Stemming/lemmatization is omitted to retain word-form
distinctions and keep a minimal deterministic pipeline; the brief makes it
conditional. The dictionary is fitted on training only. Empty processed text
maps to UNK: 22 train, one validation and zero test reviews. Raw malformed/blank
counts are zero and all splits are balanced. Distribution/OOV/truncation and
duplicate audits are in `outputs/full/`. Seven raw-text hashes are shared by
train/test; official rows are unchanged. This is not a duplicate-clean benchmark.

The tuner verifies all split/cache fingerprints including test metadata/arrays,
then removes the test dataset before training. Only train/validation tensors
reach training; no new test predictions/metrics inform selection. This does
not claim test-file bytes were never read for integrity verification.

## Reproduce and inspect

Use the pinned isolated environment from the repository root. The
[finalization guide](../../PART2_FINALIZATION.md) records setup, exact recipes,
resume, evaluation, packaging and the visible Windows log viewer. After setup:

```powershell
.venv\Scripts\python.exe -m lab1.run --task sentiment --mode smoke --device cpu
.venv\Scripts\python.exe scripts/run_sentiment_desktop.py --output runs/part2-new-full --stage all --device cuda
```

Smoke is synthetic and establishes execution only. Full reproduction needs
verified public-data/encoded caches; ignored datasets are not supplied by a
Git clone. The runner freezes source/config/recipes, trains each family,
freezes new selection and evaluates. Use a new output for a new run. Reissuing
the same command after interruption verifies the frozen contract, retains
completed invocations and resumes unfinished ones from `last.pt`, including
RNG/within-epoch state. Do not run two writers against one active directory.
Numerical results can differ across hardware/library versions.

`src/sentiment.ipynb` has eight executed cells with visible outputs and zero
errors. Selected models passed independent CPU/CUDA reload, finite-output
and padding-invariance checks. The extracted Part 2 ZIP also passed notebook,
real-model CPU and offline-smoke checks. Receipts are in root `verification/`.

## Results, interpretation and review

[Results](results.md), `metrics_report.csv` and `outputs/full/` contain
accuracy; macro/micro/weighted precision, recall and F1; confusion matrices;
ROC-AUC; trapezoidal PR-AUC; MCC; Brier; 15-bin top-label ECE; 1,000-draw IID
test-row bootstrap 95% intervals for accuracy/macro-F1/MCC; two unadjusted
exact paired McNemar tests against MLP; and predefined length/negation/OOV
slice support, macro-F1 and error rate. Intervals describe test-sample variation,
not training-seed variation. Undefined single-class values remain null.

BiLSTM improves test accuracy by 2.93 percentage points over MLP and 0.49 over
CNN in this single-seed replication, with substantially more measured training
time. CNN has lowest ECE (0.01276); BiLSTM has lowest Brier (0.02980). These
measure different calibration properties. Paired baseline comparisons favor
both experiments; they do not replace seed replication or prove new-domain
robustness. GPU memory is allocator peak, host RSS is process-lifetime peak,
and training/evaluation host peaks are separate. Epoch training time excludes
setup/validation/export; provenance records broader invocation elapsed time.
These recorded costs are not a controlled speed benchmark.

Each model has 20 actual errors: five confident FP, five confident FN, five
near-threshold and five predefined long-review slice cases. Separate
`ai_error_review_draft.csv` files and [failure analysis](failure_analysis.md)
give 60 AI-assisted types, exact quotes, explanations and one future testable
fix per case, bound to example/checkpoint/source-text hashes. Original human
packets remain unchanged, with blank `error_type`/`testable_fix` and false
`student_reviewed`. Manual review remains pending. Selected cases cannot
estimate error-category prevalence. Interpretations are hypotheses; future
fixes require training/validation studies and appropriate new evaluation,
without tuning to these test errors.

The [requirement mapping](../REQUIREMENTS_CHECKLIST.md) separates verified
technical artifacts from student design/analysis/viva, teammate independence,
team comparison and combined-report obligations. See [AI disclosure](../../AI_USE.md).
`dist/Part2_Srinidhi_2342.zip` includes both best/last weights and processed
data. Git publication uses `srinidhi/part2-desktop-final` with selected best
weights tracked through targeted exceptions; the final handoff and `git ls-remote`
establish the actual remote commit before a clone is used to restore them. Canvas still
requires one combined three-part ZIP and `Report.pdf` with the repository link.
