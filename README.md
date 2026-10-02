# DATA266 Lab 1 — Lab Pair 49

Shared repository: https://github.com/Mrnidhi/Data266_Lab

Each member independently implements and trains all three tasks. Members keep code, configurations, preprocessing, outputs and write-ups in their own named folder under each task. The team agrees on an evaluation protocol and submits one combined report.

## Current status

Complete selected-model offline backups, including processed data and checkpoint
files, are available through Git LFS for [Parts 1 and 2](reproducibility/packages/parts1-2-20261002/README.md)
and [Part 3](reproducibility/packages/part3-20261002/README.md).

The October 1 quality study selected the larger GPT and wider MLP, retaining
the original BiLSTM and CNN. The preceding desktop baseline remains in commit
`87f00dc` and separate baseline ZIPs. These verified results are on `main`;
teammate branches may contain ongoing independent work.

Srinidhi's **Part 1** trained from scratch on the desktop RTX 5090 using 100,000 training stories and 10,000 validation stories for 16 epochs and 28,416 updates. The selected epoch-16 model has 6 blocks, 8 heads, width 256 and a 512-character context. At that context, validation cross-entropy is **0.5512**, perplexity **1.7353** and character accuracy **82.44%**. On identical 256-character windows, cross-entropy improved from **0.7211 to 0.5994** (16.88%) and accuracy from **77.31% to 80.91%**. Its evaluation-mode train–validation gap is **0.0082**; validation loss improved throughout the run. See the [comparison](verification/part1_quality_comparison.json), [results](task1_llm/srinidhi/results.md), [metrics](task1_llm/srinidhi/metrics_report.csv), [notebook](task1_llm/srinidhi/src/gpt.ipynb) and [finalization guide](PART1_FINALIZATION.md). The guide and [requirement mapping](task1_llm/srinidhi/REQUIREMENTS_CHECKLIST.md) distinguish technical evidence from student/team obligations.

Srinidhi's **Part 2** compared four new candidates with three desktop controls on the same 504,000/56,000 train/validation split. Selected test accuracy is **96.12% for BiLSTM, 95.63% for CNN and 93.31% for MLP**, using all 38,000 official test reviews. The wider CNN's validation gain was within the declared 0.001 macro-F1 tolerance, so the smaller CNN was retained. The MLP uses epoch 5 and BiLSTM epoch 11 to avoid later validation deterioration. The [eight-cell notebook](task2_sentiment/srinidhi/src/sentiment.ipynb), CPU/CUDA inference and extracted-package notebook/CPU smoke checks passed. See the [results](task2_sentiment/srinidhi/results.md), [60-case AI-assisted draft](task2_sentiment/srinidhi/failure_analysis.md), [finalization guide](PART2_FINALIZATION.md) and [requirement mapping](task2_sentiment/REQUIREMENTS_CHECKLIST.md). Student manual review remains pending. The official test set had been observed in earlier work; new test predictions and scores did not inform this study's selection.

Srinidhi's **current Part 3 result (October 2)** is the validation-selected epoch-four EMA continuation at LR `5e-5`, with mean validation KID **0.01400576**. The supplied class evaluator gives mean FID **99.931933**, mean MiFID **0.412837**, and local composite `(FID + MiFID) / 2` **50.172385**. Start with the [latest complete package and dataset backup](reproducibility/packages/part3-20261002/README.md), stored through Git LFS; extract the package for its executed notebook, selected weights, JPEGs, metrics and reproduction instructions. The member notebook, CSVs and `outputs/full/` currently outside that archive document the historical 30-epoch baseline. Current class **A=Monet, B=Photo**: `pred_A2B` has 300 Monet-to-Photo JPEGs and `pred_B2A` has 7,038 Photo-to-Monet JPEGs; older A/B filenames used the opposite convention. No official upload, rank or marks are claimed. Two independent ratings of the current model's fixed 30 samples, student review, teammate comparison and the combined submission remain pending.

The complete Part 3 ZIP is an immutable backup containing original machine-path provenance and a private audit mapping. Share only its separate `human_review_packet.zip` with raters. It is supporting evidence rather than a cleaned final team submission. Earlier results remain in the [historical member report](task3_gan/srinidhi/results.md) and [September continuation report](report/PART3_FINETUNING_RESULTS.md).

## Layout

```text
Data266_Lab/
├── README.md
├── task1_llm/
│   ├── data/
│   ├── srinidhi/
│   └── revanth0211/
├── task2_sentiment/
│   ├── data/
│   ├── srinidhi/
│   └── revanth0211/
├── task3_gan/
│   ├── data/
│   │   ├── monet_jpg/
│   │   └── photo_jpg/
│   ├── srinidhi/
│   └── revanth0211/
├── reproducibility/
│   ├── manifests/
│   └── raw_logs/
└── report/
    └── REPORT_TEMPLATE.md
```

Each named member folder follows the PDF:

```text
srinidhi/
├── src/                     # Task code and executed notebook
├── data_processed/          # This member's preprocessing only
├── checkpoints/
├── outputs/
├── metrics_report.csv
├── failure_analysis.md
└── results.md
```

Srinidhi's task folders additionally contain a configuration and README. The latest extracted Part 3 package includes `evaluate_local.py`, `full_metrics_report.csv`, `submission.csv`, and `outputs/pred_A2B/` / `outputs/pred_B2A/`, as shown in the PDF's GAN tree. The [package landing page](reproducibility/packages/part3-20261002/README.md) identifies the current verified class outputs and preserves the historical repo artifacts separately. The final team PDF will be `report/DATA266_Lab1_Report_Team_49.pdf` after the remaining team evidence and comparisons are complete.

Lab Pair 49 uses `srinidhi/` for [Mrnidhi](https://github.com/Mrnidhi) and `revanth0211/` for [Revanth0211](https://github.com/Revanth0211). Revanth's folders are empty contribution scaffolds; his independent implementation and results are pending. Shared runner/setup/test utilities remain in `src/`, `scripts/`, and `tests/`; execution receipts are in `verification/`. See [team workflow](TEAM_WORKFLOW.md).

## Windows setup and one-command smoke test

Use Python 3.12 and an isolated `.venv` from the repository root. The desktop run uses Python 3.12.14, the exact direct dependency versions in `requirements.txt`, and PyTorch `2.11.0+cu128` / torchvision `0.26.0+cu128`. Its CUDA 12.8 build supports the RTX 5090's `sm_120` architecture; actual hardware and installed packages are recorded in the [desktop environment receipt](verification/part1_desktop_environment.json) and the raw run's `provenance/` manifest.

```powershell
git clone https://github.com/Mrnidhi/Data266_Lab.git
cd Data266_Lab
git lfs install
git lfs pull
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
```

Install Git LFS before the two `git lfs` commands. The selected wider MLP uses
LFS because its complete checkpoint is 124,098,311 bytes. ZIPs contain the
complete checkpoint bytes.

If `py -3.12` is unavailable, use the executable for an installed Python 3.12 interpreter to create `.venv`. In VS Code select `.venv\Scripts\python.exe` as the notebook interpreter. Part 1 needs no GAN evaluation extras. Parts 2/3 reproduction additionally requires the task data and, for GAN metrics, `requirements-image-metrics.txt`.

After setup, this **single native Windows command** runs an offline synthetic Part 1 CPU smoke test and chooses a fresh output directory automatically:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode smoke --device cpu
```

Success checks execution and saved outputs; it does not establish full-data training or story quality. Outputs are saved beneath `runs/` with a fresh timestamp. See the Part 1 README for full training and finalization commands.

## Linux/macOS one-command smoke test

From a fresh clone, run:

```bash
bash scripts/smoke.sh
```

This prepares an isolated Python 3.12 environment, installs pinned dependencies and runs all five models briefly on synthetic CPU data. Initial installation needs internet; smoke training itself is offline. It does not rent a GPU or start RunPod. Outputs go into a new folder under `reproducibility/raw_logs/srinidhi/`. Success verifies execution, not model quality.

```bash
git clone https://github.com/Mrnidhi/Data266_Lab.git
cd Data266_Lab
bash scripts/smoke.sh
```

If a specific Python 3.12 executable is available, set `LAB1_PYTHON` to it. For notebooks, select this repository's `.venv` interpreter. Clone the complete repository and use the editable installation; the member source files are required alongside the runner. A standalone wheel is not the supported distribution.

## Models and configurations

| Task | Srinidhi's proposed full model | Configuration |
|---|---|---|
| Character GPT | Manual causal attention, 6 blocks, 8 heads, width 256, context 512, 16 epochs | `task1_llm/srinidhi/outputs/full/reproduction_config.json` |
| Yelp sentiment | Max-pool embedding MLP, BiLSTM, residual dilated CNN; learned embeddings | `task2_sentiment/srinidhi/outputs/full/reproduction/` |
| CycleGAN | Two 9-block generators and two PatchGAN discriminators, 256px; selected epoch-four paired EMA continuation | [Latest package recipe](reproducibility/packages/part3-20261002/README.md) |

`smoke` uses tiny synthetic inputs and smaller models. `rehearsal` uses proposed dimensions with small public text subsets and short runs; GAN inputs remain explicitly synthetic until class data are supplied. `full` selects complete training. Full GAN mode refuses missing data/manifests. Use the published reproduction recipes for the selected Parts 1/2 models; the original configuration files remain preserved.

On an already allocated Linux NVIDIA machine:

```bash
bash scripts/bootstrap_gpu.sh
.venv/bin/python scripts/prefetch_rehearsal.py
.venv/bin/python scripts/rehearsal.py --device cuda --mode rehearsal --max-minutes 60
```

See each task's member README for full training, resume and evaluation. The [compute allocation plan](COMPUTE_PLAN.md) records the desktop placement, earlier cloud costs and checkpoint transfers. The [RunPod handoff](RUNPOD_HANDOFF.md) records the earlier optional cloud workflow; the new Part 1 run uses the desktop. The scripts do not allocate or stop a pod; their time limit stops training, not billing.

For visible training progress, see the [SSH terminal workflow](SSH_WORKFLOW.md). All model runners print flushed step/loss/elapsed/estimated-time updates; the connection launcher requires the actual running Pod's SSH host/port and a registered key.

## Data, evidence and submission

Raw text datasets are cached under each task's `data/huggingface/`; Srinidhi's selected rows, tokenizer mappings and preprocessing stay in his own `data_processed/`. Verified GAN images belong in `task3_gan/data/monet_jpg/` and `photo_jpg/`, with individual split manifests under the member's `data_processed/`. Large data, environments, caches, ZIPs and most model binaries are ignored by Git. The baseline snapshot on `main` includes selected Parts 1/2 best weights. Verified final study results and selected weights are authorized for a normal push to `main`, using Git LFS for the large selected MLP. Both best/last weights, raw evidence and processed data are included in verified local Part 1/Part 2 ZIPs. Save other weights separately and document their accessible location/checksum; include required weights in the final Canvas ZIP. Never commit credentials or personal absolute paths.

Preserve raw logs/manifests without editing them after a run. For desktop Part 1, `metrics.jsonl` is untouched step/epoch evidence in the baseline Git snapshot and local ZIP. The original console `RUN_LOG.txt` remains local; startup warnings contain machine-specific paths, so this file is excluded from Git and the ZIP. The October 1 Part 1 replacement was explicitly requested; it does not establish recovery of missing older checkpoints. New published evidence uses portable relative paths, with hashes linking derived copies to preserved raw originals. Complete Part 2 raw/data evidence remains available locally and in verified archives; compact verified final study evidence can be published to `main`. Each member must complete `results.md`, `failure_analysis.md` and `metrics_report.csv` using actual evidence. Agree on the evaluation protocol before comparing models.

Finish `report/DATA266_Lab1_Report_Team_49.pdf`, include this repository link, compare all members and package the final Canvas submission as one ZIP containing separate Part 1, Part 2 and Part 3 folders plus the combined `Report.pdf`. Each part needs an executed notebook with visible outputs, actual model weights and applicable outputs. Individual Part 1/Part 2 archives do not complete the whole submission. Student manual error review/viva preparation, Revanth's independent results, team comparisons, the combined report and the remaining Part 3 requirements remain pending. See [submission checklist](SUBMISSION_CHECKLIST.md), [report outline](report/REPORT_TEMPLATE.md), [research notes](RESEARCH_NOTES.md), and [AI assistance disclosure](AI_USE.md).
