# Part 2 — fresh desktop reproduction and finalization

Repository: https://github.com/Mrnidhi/Data266_Lab

This guide describes Srinidhi's fresh October 1, 2026 desktop reproduction of
the three selected Yelp Polarity sentiment recipes. Training uses an Intel
Core Ultra 9 285K CPU and NVIDIA GeForce RTX 5090 with Python 3.12.14,
PyTorch 2.11.0+cu128 and CUDA 12.8. The hardware/package receipt is
`verification/part2_desktop_environment.json`.

Finalization creates `dist/Part2_Srinidhi_2342.zip` at the producing repository
root, containing one `Part 2/` folder. After successful creation, extract and
open that folder as the working directory; its files preserve repository-relative paths.
The desktop run is
`reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/`.
Its `desktop_plan.json` freezes runtime source hashes, base configuration and
three recipe files before training. Each family starts from random weights
and learns its own embeddings. The run is a replication of previously chosen
architectures and schedules, not a new hyperparameter search.

The full data and encoded cache have been prepared, original rows verified,
and CUDA smoke/preflight checks passed. Fresh full training is underway.
Completed training, selected checkpoints, final test metrics, notebook and
submission package must be established by the actual new receipts and outputs;
the earlier cloud numbers are preserved as historical evidence.

## Windows setup and one-command smoke test

Work from the repository root with Python 3.12 and an isolated `.venv`:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
```

The project requirements record exact direct dependency versions. The standalone
archive uses a generated Part 2 environment file and package metadata so its
setup needs only this task's runtime. If `py`
is unavailable, create `.venv` with an installed Python 3.12 executable.
Choose `.venv\Scripts\python.exe` as the VS Code notebook interpreter.
The RTX 5090 requires a compatible CUDA PyTorch build; the desktop's tested
CUDA 12.8 build supports its `sm_120` architecture. Part 2 does not need GAN
evaluation packages from `requirements-image-metrics.txt`.

After setup, one offline CPU command checks all three sentiment models using
small synthetic data and a new timestamped output directory:

```powershell
.venv\Scripts\python.exe -m lab1.run --task sentiment --mode smoke --device cpu
```

Smoke results establish execution, not Yelp accuracy or full training.
On Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`
after creating an equivalent Python environment.

## Fixed data and recipes

The official dataset is `fancyzhx/yelp_polarity`, resolved revision
`bbf1c97a1f0cf005e5aded43839fd814654a1557`. Its 560,000 training rows are
stratified into 504,000 training and 56,000 validation rows with seed 2342.
All 38,000 official test rows are reserved for final evaluation. The new
prepared data match historical frozen row contents and ordering exactly;
`verification/part2_desktop_data_identity.json` records both desktop byte
hashes and normalized-LF hashes because the desktop JSONL files use CRLF.

Full JSONL files and manifests are under
`task2_sentiment/srinidhi/data_processed/full/`; encoded features are under
`task2_sentiment/srinidhi/data_processed/full_encoded/`. The cache verifies
content, configuration, preprocessing and vocabulary. Copies must retain
their complete manifests and matching files. A Git clone does not supply
these ignored dataset caches; prepare the public data or obtain the verified
processed cache separately before full training.

The common recipe has vocabulary cap 40,000, minimum frequency 2, learned
embedding width 128, maximum length 384 tokens, batch 128, AdamW weight
decay 0.0001, gradient clipping 1.0 and validation-based learning-rate decay.
Lowercasing/casefolding, HTML/contraction handling, regex word tokenization
and a customized stopword list preserve negation and contrast words. The
vocabulary is fitted on training rows only. Stemming/lemmatization is not
implemented; the brief makes it conditional and the student must explain
the preprocessing choices.

| Model | Architecture | Schedule |
|---|---|---|
| Baseline `maxpool_mlp` | Embedding → masked max pool → dense 64/ReLU/dropout → binary logit | LR 0.0015, maximum 6 epochs, early-stopping patience 2 |
| Experimental `bilstm` | One bidirectional LSTM, 96 units per direction → masked max pool → dense 64/dropout → logit | LR 0.001, maximum 12 epochs, patience 4 |
| Experimental `dilated_cnn` | 128 channels, four residual convolution blocks, kernel 3, dilations 1/2/4/8 → masked pool → dense 64/dropout → logit | LR 0.0008, maximum 6 epochs, patience 2 |

Embedding/head dropout is 0.3 and CNN-block dropout is 0.2. Windows training
uses `num_workers=0`. Each checkpoint updates only when validation macro-F1
improves by more than 0.0001, following the existing selection/early-stopping
rule; its selected epoch is not necessarily the strict numerical maximum
over all epoch scores.

The historical test results and error texts have already been inspected.
The repeated official test set is therefore not a newly sealed holdout.
New checkpoint selection uses validation only, and a new manifest freezes
checkpoint paths/hashes before any new test predictions are calculated.
Do not use new test scores or error examples to choose further recipes.

The plan's “no new test access during training” shorthand concerns test
inference, predictions and metrics. Before training, the tuner validates all
frozen split/cache fingerprints, including test-row metadata and encoded
arrays, then removes the test dataset from the model's training call. The
training function receives only train/validation tensors; this workflow does
not claim that test-file bytes were never read for integrity checks.

## Run, resume and evaluate

The recorded desktop command is:

```powershell
.venv\Scripts\python.exe scripts/run_sentiment_desktop.py --output reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full --stage all --device cuda
```

For a separate reproduction choose a fresh output, for example:

```powershell
.venv\Scripts\python.exe scripts/run_sentiment_desktop.py --output runs/part2-new-full --stage all --device cuda
```

To view live output for all three training logs and the selected evaluation,
run this separate terminal viewer from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/watch_sentiment_desktop.ps1
```

Closing the viewer does not stop the training job. The viewer follows the
existing logs; the recorded runner command controls training.

The runner freezes the plan, trains the three families sequentially with only
train/validation tensors passed to training, freezes `selection_manifest.json`, and evaluates
the selected weights into `selected/`. Its stages are `plan`, `train`,
`evaluate` and `all`. Recipe files are saved in the run's `recipes/` folder;
invocation provenance is in `training/<model>/provenance.json`; model histories
and checkpoints are in `training/<model>/<model>/` beneath that run. The
nested model name reflects the shared single-candidate training interface.

Do not run two copies against the same active output. After an interruption,
the same `--stage all` command checks frozen source/configuration/recipe
hashes, keeps completed training invocations and resumes unfinished ones from
their `last.pt`. Each checkpoint includes model/optimizer/scheduler/AMP state,
vocabulary, configuration, RNG/shuffle state and training progress. Preserve
the complete run and best/last files together; changed training contracts or
source hashes are rejected. Cross-environment floating-point results may
differ even when data, seed and resume state match.

To evaluate a completed, already frozen selection independently:

```powershell
.venv\Scripts\python.exe scripts/finalize_sentiment_selection.py --selection reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/selection_manifest.json --output runs/part2-repeat-evaluation --device cuda
```

The final evaluator verifies actual source/checkpoint/data identity and
requires matching test IDs/labels across all three models. It calculates
all required metrics, bootstrap intervals, slice results and paired McNemar
tests. Using the ordinary six-epoch reference `--mode full` profile alone
does not reproduce the selected twelve-epoch BiLSTM schedule.

## Publish evidence and complete manual review

Once desktop training and the final selected evaluation have completed, the
end-to-end finalization command publishes the results, executes the notebook,
checks saved inference and builds the verified Part 2 archive:

```powershell
.venv\Scripts\python.exe scripts/finalize_part2.py --run-dir reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/selected --data-dir task2_sentiment/srinidhi/data_processed/full
```

The optional `--device cpu` supports finalization without CUDA. It changes
finalization/inference placement, not the recorded original training hardware.
The export-only command is:

```powershell
.venv\Scripts\python.exe scripts/publish_sentiment_results.py --run-dir reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/selected --data-dir task2_sentiment/srinidhi/data_processed/full
```

Publication verifies source/checkpoint mappings and preserves previous
publication bytes under `task2_sentiment/srinidhi/outputs/publication_history/`
before replacing the current result files. Old annotations and review sets
must stay traceable to their own checkpoint and predictions. Portable
history/provenance evidence is copied byte-for-byte; original console logs
containing machine-specific startup paths remain local without editing.

The new member folder must contain `metrics_report.csv`, `results.md`, an
executed `src/sentiment.ipynb`, actual selected weights and required outputs.
The notebook must show actual new results and independently reload the saved
weights. The package's verification record must identify the exact selected
files, original training runs, completed notebook and archive hashes.

Receipts are `verification/part2_export.json`, `part2_notebook.json`,
`part2_checkpoint_inference.json` and `part2_finalization.json`. The producing
repository stores the archive CRC/inventory/content-hash record in
`dist/part2_package_verification.json` and the whole-file checksum in
`dist/Part2_SHA256SUMS.txt`. Verify a transferred ZIP against that checksum
and rerun its content checks before relying on the copy:

```powershell
Get-FileHash -Algorithm SHA256 dist/Part2_Srinidhi_2342.zip
.venv\Scripts\python.exe scripts/finalize_part2.py --verify-archive dist/Part2_Srinidhi_2342.zip
```

After initial finalization, adding separate AI drafts or final documentation
does not require retraining, another test evaluation or republication. Rebuild
only the archive while checking the existing metric/notebook/inference proof:

```powershell
.venv\Scripts\python.exe scripts/finalize_part2.py --run-dir reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/selected --data-dir task2_sentiment/srinidhi/data_processed/full --archive-only
```

This preserves the original training, evaluation, dataset, runtime source and
executed-notebook bytes and includes the latest member documentation and
separate AI draft CSVs. Its receipt is `verification/part2_archive_only.json`.

All required numeric measures are listed in
`task2_sentiment/REQUIREMENTS_CHECKLIST.md`: accuracy; macro/micro/weighted
precision, recall and F1; confusion matrix; ROC/PR-AUC; MCC; Brier/ECE;
bootstrap intervals; paired McNemar comparisons; slice macro-F1/error rate;
parameter count, training time, throughput and peak memory. Metric definitions
and uncertainty/measurement limits must accompany the numbers.

For **each** model, `required_20_errors_for_review.csv` must contain five
confident false positives, five confident false negatives, five near-threshold
errors and five slice-specific failures. The current pipeline uses long
reviews as the predefined slice and avoids duplicate IDs within each model.
Each case needs an error type, explanation grounded in the full review and
one testable fix. This creates 60 model-specific entries, not necessarily
60 unique review texts across models.

AI-assisted classifications belong only in clearly named `ai_draft_*`
columns or labeled prose. Leave `error_type`, `testable_fix` and
`student_reviewed` unchanged until the student actually reviews each case.
AI drafts do not complete the required manual analysis; no human review is
fabricated. Suggested fixes are future validation studies, not test-set
tuning instructions.

## Submission scope

A final Part 2 archive must include the executed notebook, actual model
weights, preprocessing/outputs and reproducibility evidence, with verified
CRC/inventory/SHA-256 values and an independently retained copy. A remote
GitHub commit must be verified before claiming the publication is pushed;
local files and app access alone are not proof of a remote update.

The course also requires each student's own core design/analysis and viva
understanding. Assisted implementation and automatically executed training
do not establish those requirements. Student review, independently trained
teammate models, team comparisons and the combined report remain separate.

Canvas asks for one final ZIP with separate Part 1, Part 2 and Part 3 folders
and one combined `Report.pdf` including the GitHub link. A Part 2-only archive
does not complete that overall submission. Preserve Part 1 and Part 3
artifacts and human-rating provenance while finalizing this part.
