# Part 2 — reproduce and finalize

Repository: https://github.com/Mrnidhi/Data266_Lab

The current evaluation is `reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected/`. It combines the new wider MLP with the retained desktop BiLSTM and CNN. The preceding publication remains under `task2_sentiment/srinidhi/outputs/publication_history/026926f7d98246d3/`; the baseline ZIP remains separate.

| Model | Completed / selected epoch | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| MLP | 9 / 5 | 0.928285 | 93.310526% | 0.933105 |
| BiLSTM | 12 / 11 | 0.958857 | 96.121053% | 0.961210 |
| CNN | 6 / 6 | 0.953446 | 95.626316% | 0.956262 |

The MLP uses embedding width 256, hidden width 128 and dropout 0.5; it stopped after 9 of a maximum 12 epochs. The other models retain embedding 128 and their recorded schedules. The frozen rule prefers fewer parameters within 0.001 of each family's highest validation macro-F1. The wider CNN's 0.000733 gain is inside that tolerance, so the original CNN is retained. All seven candidates and settings are in the published selection and training-source manifests.

Fresh metric/source checks, notebook execution, saved-model inference and extracted-package checks passed for this selected suite. The extracted notebook ran all eight cells without errors, and the offline CPU smoke test completed all three models. The final ZIP is rebuilt after documentation updates and checked by content hashes. Student review and teammate/report obligations remain separate.

## Setup and inspect

Use Python 3.12 from the repository root, or the extracted ZIP's `Part 2/` folder. Recorded training used Python 3.12.14, PyTorch 2.11.0+cu128 and CUDA 12.8 on an Intel Core Ultra 9 285K / RTX 5090.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m lab1.run --task sentiment --mode smoke --device cpu
```

The standalone ZIP has its own pinned Part 2 dependencies. Smoke uses synthetic data and checks execution only. On Linux/macOS use `.venv/bin/python`. Select this environment's kernel for `task2_sentiment/srinidhi/src/sentiment.ipynb`.

After installing Git LFS, clone users should run:

```sh
git lfs install
git lfs pull
```

LFS applies specifically to `task2_sentiment/srinidhi/checkpoints/maxpool_mlp/best.pt`: the full file is 124,098,311 bytes. Compare it with `task2_sentiment/srinidhi/checkpoints/manifest.json`; a pointer is not loadable. Selected best weights are intended for the authorized normal push to `main`. The local ZIP packages full real best/last files, data and raw evidence, including files ignored by Git.

The cache contains 504,000 training, 56,000 validation and 38,000 test reviews. Reproduction requires `data_processed/full/` and `full_encoded/` with their manifests; Git alone does not supply them. Vocabulary is fitted on training only. Each model learns its own embedding and uses the first 384 processed tokens.

## Reproduce selected models

Use new output directories and the actual published recipes:

```powershell
.venv\Scripts\python.exe scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/maxpool_mlp_candidate.json --output runs/reproduce-part2-maxpool_mlp --device cuda
.venv\Scripts\python.exe scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/bilstm_candidate.json --output runs/reproduce-part2-bilstm --device cuda
.venv\Scripts\python.exe scripts/tune_sentiment.py --candidate task2_sentiment/srinidhi/outputs/full/reproduction/dilated_cnn_candidate.json --output runs/reproduce-part2-dilated_cnn --device cuda
```

Use the configured validation checkpoint rule and preserve best/last files, histories and provenance. Do not run two writers against one output. Resume from matching `last.pt` only under the same frozen training contract. Different hardware/library versions can change numerical results.

To evaluate the **existing preserved selection**, use:

```powershell
.venv\Scripts\python.exe scripts/finalize_sentiment_selection.py --selection reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected/selection_manifest.json --output runs/reproduce-part2-final-evaluation --device cuda
```

This needs complete local source runs or the extracted ZIP. It references preserved weights, not newly retrained ones. New runs need their own validation-selected manifest before test evaluation. Earlier test scores/texts were observed; do not choose further recipes from these results. Integrity checks read all cache fingerprints, but only training/validation tensors reach training.

## Finalize and verify

After selected evaluation and current error drafts are ready, run:

```powershell
.venv\Scripts\python.exe scripts/finalize_part2.py --run-dir reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected --data-dir task2_sentiment/srinidhi/data_processed/full
```

Use `--device cpu` when CUDA is unavailable. Finalization verifies metrics and source identities, publishes results, executes the notebook, checks saved-model inference and creates `dist/Part2_Srinidhi_2342.zip`. Publication preserves prior results; annotations stay linked to their own model and predictions. The ZIP contains one `Part 2/` folder with full checkpoint bytes and processed data, plus compact candidate evidence without unselected weights or recursive history.

Check current receipts: `verification/part2_export.json`, `part2_notebook.json`, `part2_checkpoint_inference.json` and `part2_finalization.json`. Confirm the selected suite path and checkpoint hashes before relying on them. Verify the ZIP and extracted notebook/CPU smoke:

```powershell
Get-FileHash -Algorithm SHA256 dist/Part2_Srinidhi_2342.zip
.venv\Scripts\python.exe scripts/finalize_part2.py --verify-archive dist/Part2_Srinidhi_2342.zip
.venv\Scripts\python.exe scripts/finalize_part2.py --verify-portability dist/Part2_Srinidhi_2342.zip
```

Compare the whole-file hash with `dist/Part2_SHA256SUMS.txt`. Content inventory and CRC/hash checks are in `dist/part2_package_verification.json`; extracted execution is in `verification/part2_package_portability.json`. Keep a verified backup.

After successful full finalization, a documentation-only package update can use saved evidence checks without retraining or reevaluating:

```powershell
.venv\Scripts\python.exe scripts/finalize_part2.py --run-dir reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected --data-dir task2_sentiment/srinidhi/data_processed/full --archive-only
```

This requires unchanged frozen data, runtime, results and notebook. Its receipt is `verification/part2_archive_only.json`.

## Complete Lab 1 deliverables

The [checklist](task2_sentiment/REQUIREMENTS_CHECKLIST.md) covers accuracy; macro/micro/weighted precision, recall and F1; confusion matrix; ROC/PR-AUC; MCC; Brier/ECE; bootstrap intervals; McNemar tests; slices; parameters, training time, examples/second and memory. Intervals describe test-row variation, not seed variation. Training costs describe actual invocations and workload overlap, not isolated architecture speed. The duplicate audit records seven shared train/test text hashes.

Each model needs 20 reviewed errors: five confident false positives, five confident false negatives, five near-threshold errors and five long-review failures. AI drafts provide exact quotes, short hypotheses and future training/validation fixes; all 60 human flags remain false. Students must inspect full reviews and record their conclusions before marking review complete. `scripts/render_sentiment_error_drafts.py` checks draft linkage and refreshes combined failure analysis while preserving human fields and notebook bytes.

A normal final push to `main` is authorized; preserve history. Complete independent teammate comparisons, the joint report and individual viva understanding honestly under `AI_USE.md`. Canvas requires one combined ZIP with Part 1, Part 2 and Part 3 folders and a combined `Report.pdf` containing the GitHub link. This Part 2 package alone does not complete submission.
