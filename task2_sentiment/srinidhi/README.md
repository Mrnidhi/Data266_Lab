# Part 2 — Yelp sentiment, Srinidhi

The current `outputs/full/` publication compares three independently trained classifiers with embeddings learned from scratch. The selected suite is `reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected/`. Its wider MLP replaces the previous MLP; BiLSTM and CNN retain their earlier desktop checkpoints.

| Model | Completed / selected epoch | Validation macro-F1 | Test accuracy | Test macro-F1 | Parameters |
|---|---:|---:|---:|---:|---:|
| Max-pool MLP | 9 / 5 | 0.928285 | 93.310526% | 0.933105 | 10,273,025 |
| BiLSTM | 12 / 11 | 0.958857 | 96.121053% | 0.961210 | 5,305,985 |
| Dilated CNN | 6 / 6 | 0.953446 | 95.626316% | 0.956262 | 5,539,073 |

Selection considered three existing controls and four completed new candidates. The frozen rule prefers fewer parameters within 0.001 of each family's highest validation macro-F1. The wider CNN reached 0.954179 versus the original CNN's 0.953446, a gain of 0.000733; the original CNN was therefore retained. Test scores did not enter this decision. Full candidate records and hashes are in [selection_manifest.json](outputs/full/selection_manifest.json).

## Models and data

| Model | Selected architecture | Initial LR / maximum epochs / patience |
|---|---|---|
| MLP baseline | Embedding 256 → masked max pool → dense 128/ReLU/dropout 0.5 → binary logit | 0.0015 / 12 / 4 |
| BiLSTM experiment | Embedding 128 → bidirectional LSTM, 96 units per direction → masked max pool → dense 64/dropout 0.3 → logit | 0.001 / 12 / 4 |
| CNN experiment | Embedding 128 → 128 channels → four residual blocks, two kernel-3 convolutions each, dilations 1/2/4/8 → masked pool → dense 64/dropout 0.3 → logit | 0.0008 / 6 / 2 |

The MLP measures unordered lexical evidence; BiLSTM adds sequence context; CNN composes local patterns over a wider receptive field. The wider MLP changes both capacity and regularization, so its improvement does not isolate one cause. Each model has its own embedding; CNN-block dropout is 0.2. Padding is masked during pooling and convolution.

All use seed 2342, a training-only vocabulary cap of 40,000, minimum frequency 2, the first 384 processed tokens, batch 128, AdamW weight decay 0.0001 and clipping 1.0. Learning rate falls on a validation-loss plateau. Checkpoint improvement must exceed 0.0001 validation macro-F1. [training_sources.json](outputs/full/training_sources.json) and each model's `training_config.json` record exact settings and CPU/GPU.

The frozen data contain 504,000 training, 56,000 validation and 38,000 test reviews, with balanced labels. Preprocessing casefolds, handles HTML/contractions, removes punctuation and selected stopwords, and retains negation/contrast. Stemming is omitted to retain word forms. Empty processed text maps to UNK: 22 training rows, one validation row and no test rows. See [data details](data_processed/README.md).

## Read and reproduce

Start with [results.md](results.md), [metrics_report.csv](metrics_report.csv) and [sentiment.ipynb](src/sentiment.ipynb). The [finalization guide](../../PART2_FINALIZATION.md) gives setup, selected-recipe training, evaluation and package checks. After setup:

```powershell
.venv\Scripts\python.exe -m lab1.run --task sentiment --mode smoke --device cpu
```

Smoke uses synthetic data to check execution. Real reproduction requires frozen processed data and complete checkpoints. The wider MLP's exact Git LFS path is `task2_sentiment/srinidhi/checkpoints/maxpool_mlp/best.pt`; after cloning or pulling, run `git lfs install` and `git lfs pull`. See [checkpoint details](checkpoints/README.md).

Metric/source checks, the eight-cell notebook, saved-model inference and the extracted package's notebook/CPU smoke test passed. The ZIP contains real best/last weights and processed data; its checksum and inventory are recorded in `dist/part2_package_verification.json`. A normal final push to `main` is authorized, preserving history.

## Required analysis and review

The publication includes classification metrics, ROC/PR curves, MCC, Brier score, 15-bin ECE, 1,000-draw bootstrap intervals, two exact paired McNemar comparisons against MLP, predefined length/negation/OOV slices, parameter counts, training time, examples/second and memory. Definitions and measurement limits are in [results.md](results.md).

The test set has previously been observed, seven text hashes are shared between train and test, and only one training seed is represented. These results do not establish a new unseen holdout, a global optimum or production readiness. Training costs come from actual invocations; differing workload overlap limits direct speed comparisons.

Each model has 20 real errors: five confident false positives, five confident false negatives, five near-threshold errors and five long-review errors. Separate AI draft CSVs support [failure_analysis.md](failure_analysis.md) with exact quotes and future testable fixes. All 60 human review flags are currently false. Student review, independent teammate comparisons and the combined report remain required; see the [checklist](../REQUIREMENTS_CHECKLIST.md) and [AI disclosure](../../AI_USE.md).

The preceding desktop publication remains under `outputs/publication_history/026926f7d98246d3/`; older results remain historical. Canvas requires the combined three-part ZIP and `Report.pdf`, not this Part 2 package alone.
