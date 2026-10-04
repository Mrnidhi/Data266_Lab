# Part 2 — Yelp sentiment, Srinidhi

Three independently trained classifiers with embeddings learned from scratch.
The current suite retains the desktop BiLSTM/CNN and selects a wider max-pool MLP.

## Results and selection

| Model | Completed / selected epoch | Validation macro-F1 | Test accuracy | Test macro-F1 | Parameters |
|---|---:|---:|---:|---:|---:|
| Max-pool MLP | 9 / 5 | 0.928285 | 93.310526% | 0.933105 | 10,273,025 |
| BiLSTM | 12 / 11 | 0.958857 | 96.121053% | 0.961210 | 5,305,985 |
| Dilated CNN | 6 / 6 | 0.953446 | 95.626316% | 0.956262 | 5,539,073 |

Seven candidates were considered. The frozen rule prefers fewer parameters within
0.001 of each family's highest validation macro-F1. The wider CNN's 0.000733 gain
was inside that tolerance, so the original CNN was retained. Test scores did not
enter this decision. [Selection](outputs/full/selection_manifest.json) and
[training sources](outputs/full/training_sources.json) record recipes and hashes.

## Models and data

| Model | Architecture | Initial LR / maximum epochs / patience |
|---|---|---|
| MLP | Embedding 256 → masked max pool → dense 128/ReLU/dropout 0.5 → logit | 0.0015 / 12 / 4 |
| BiLSTM | Embedding 128 → BiLSTM, 96 units/direction → masked max pool → dense 64/dropout 0.3 → logit | 0.001 / 12 / 4 |
| CNN | Embedding 128 → 128 channels → four residual blocks with kernel 3, dilations 1/2/4/8 → masked pool → dense 64/dropout 0.3 → logit | 0.0008 / 6 / 2 |

Each model learns its own embedding. CNN blocks use two convolutions and dropout
0.2. Padding is masked. All use seed 2342, vocabulary cap 40,000, minimum frequency
2, token cap 384, batch 128, AdamW weight decay 0.0001 and clipping 1.0. Learning
rate falls on a validation-loss plateau; checkpoint macro-F1 must improve by
0.0001. The wider MLP changes capacity and regularization together.

The frozen data contain 504,000 training, 56,000 validation and 38,000 test reviews
with balanced labels. Vocabulary is training-only. Preprocessing casefolds,
handles HTML/contractions and removes punctuation/selected stopwords, retaining
negation and contrast. Stemming is omitted. Empty processed inputs map to UNK
(22 training, one validation, zero test). See [data details](data_processed/README.md).

## Setup and reproduction

Work from the repository root or extracted ZIP's `Part 2/` folder. Use Python 3.12;
recorded training used PyTorch 2.11.0+cu128 / CUDA 12.8 on an RTX 5090.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m lab1.run --task sentiment --mode smoke --device cpu
```

On Linux/macOS use `python3.12` and `.venv/bin/python`. Smoke checks synthetic CPU
execution only. Select this environment's kernel for [sentiment.ipynb](src/sentiment.ipynb).
After cloning, run `git lfs install` and `git lfs pull` for the 124,098,311-byte MLP
checkpoint. A pointer is not loadable; compare [checkpoint hashes](checkpoints/manifest.json).
The standalone ZIP contains real best/last weights, `data_processed/full/` and
`full_encoded/` with manifests; a Git clone alone does not supply the data caches.

Run each selected recipe into a new output directory:

```powershell
.venv\Scripts\python.exe scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/maxpool_mlp_candidate.json --output runs/reproduce-part2-mlp --device cuda
.venv\Scripts\python.exe scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/bilstm_candidate.json --output runs/reproduce-part2-bilstm --device cuda
.venv\Scripts\python.exe scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/dilated_cnn_candidate.json --output runs/reproduce-part2-cnn --device cuda
```

Keep matching configs, best/last states and histories. Resume only under the same
training contract; never use two writers per output. Newly trained candidates
need a new validation-only selection before test evaluation. The saved suite can
be reevaluated with `scripts/finalize_sentiment_selection.py --selection <saved-selection.json> --output <new-evaluation-dir> --device cuda`.

## Verify and package

```powershell
.venv\Scripts\python.exe scripts/finalize_part2.py --run-dir reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected --data-dir task2_sentiment/srinidhi/data_processed/full
.venv\Scripts\python.exe scripts/finalize_part2.py --verify-archive dist/Part2_Srinidhi_2342.zip
.venv\Scripts\python.exe scripts/finalize_part2.py --verify-portability dist/Part2_Srinidhi_2342.zip
```

Add `--device cpu` when CUDA is unavailable. Finalization verifies metrics/source
identities, executes the notebook, checks inference and creates the ZIP without
training. After a successful finalization, `--archive-only` rebuilds documentation
using saved checks only if frozen data, runtime, results and notebook are unchanged.
Receipts are `verification/part2_*.json`; ZIP inventory, CRC and SHA-256 are in
`dist/part2_package_verification.json` and `dist/Part2_SHA256SUMS.txt`.

## Analysis and remaining review

[Results](results.md) and [metrics](metrics_report.csv) include classification
metrics, ROC/PR, MCC, Brier/ECE, bootstrap intervals, McNemar comparisons, slices,
parameters, time, throughput and memory. One seed, seven shared train/test text
hashes and previously observed test data limit generalization claims. Intervals
measure test-row variation; overlapping GPU work limits speed comparisons.

Each model has 20 real errors across confident false positives/negatives,
near-threshold and long-review groups. [Failure analysis](failure_analysis.md)
contains AI-assisted drafts; all 60 human flags remain false. Review full inputs
before marking them complete. `scripts/render_sentiment_error_drafts.py` verifies
draft linkage and updates prose while preserving human fields and notebook bytes.
See [requirements](../REQUIREMENTS_CHECKLIST.md),
[assistance](../../README.md#contributions-and-assistance) and
[submission](../../README.md#submission). Team comparison and report remain separate.
