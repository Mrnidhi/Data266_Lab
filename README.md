# DATA266 Lab 1 — Lab Pair 49

Shared repository: https://github.com/Mrnidhi/Data266_Lab

Each member independently implements and trains all three tasks. Members keep code, configurations, preprocessing, outputs and write-ups in their own named folder under each task. The team agrees on an evaluation protocol and submits one combined report.

## Current status

Srinidhi's **Part 1 full training completed from scratch on the Windows desktop RTX 5090** on October 1, 2026: 100,000 training stories, 10,000 validation stories, 12 complete epochs and 18,732 updates. The selected epoch-12 checkpoint achieved validation cross-entropy **0.7211**, character perplexity **2.0567** and next-character accuracy **77.31%**. The user requested removal and replacement of the previous Part 1 runs; these numbers come only from the new desktop run. Its untouched evidence lives in `reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full/`. The [results](task1_llm/srinidhi/results.md), [38-row metric report](task1_llm/srinidhi/metrics_report.csv), [notebook](task1_llm/srinidhi/src/gpt.ipynb) and actual selected weights are published in the member folder. The [finalization guide](PART1_FINALIZATION.md) documents setup, inference/notebook checks and the portable `dist/Part1_Srinidhi_2342.zip`; [the requirement mapping](task1_llm/srinidhi/REQUIREMENTS_CHECKLIST.md) identifies the remaining student/team obligations.

Srinidhi's **Part 2 fresh desktop reproduction also completed on October 1**. Three separately initialized models trained on the same 504,000/56,000 train/validation split; checkpoints were frozen from validation before new inference on all 38,000 official test reviews. The BiLSTM achieved **96.12%** test accuracy, CNN **95.63%**, and MLP **93.19%**, selecting epochs 11, 6 and 5 respectively. These results come from the Intel Core Ultra 9 285K / RTX 5090 desktop, not the older cloud/Mac run. The [notebook](task2_sentiment/srinidhi/src/sentiment.ipynb) has eight executed cells with outputs and no errors; saved models passed CPU/CUDA reload and padding checks, and the extracted Part 2 package passed notebook and offline smoke checks. The [results](task2_sentiment/srinidhi/results.md), [60-case AI-assisted draft](task2_sentiment/srinidhi/failure_analysis.md), [finalization guide](PART2_FINALIZATION.md) and [requirement mapping](task2_sentiment/REQUIREMENTS_CHECKLIST.md) record technical evidence and pending manual review. Earlier nine-candidate cloud selection is preserved in publication history and [research notes](task2_sentiment/srinidhi/RESEARCH_AND_SEARCH.md); the desktop run replicates three fixed recipes and the official test set had already been observed.

Part C baseline training completed on one RTX 5090: 30 epochs and 168,720 updates with zero non-finite events. The validation-selected step-168,720 checkpoint was evaluated on both frozen test directions. Photo-to-Monet achieved KID 0.01111 and content cosine 0.8103; Monet-to-photo achieved KID 0.01920 and content cosine 0.9019. See the baseline [executed notebook](task3_gan/srinidhi/src/cyclegan.ipynb), [results](task3_gan/srinidhi/results.md), [metrics](task3_gan/srinidhi/metrics_report.csv), and [failure-analysis draft](task3_gan/srinidhi/failure_analysis.md). The full run and evidence archives were hash-verified locally before Vast instance 51639909 was stopped. The later September 29 continuation, its selected results and limitations are in [Part 3 continuation results](report/PART3_FINETUNING_RESULTS.md). One personal human review is recorded; a second independent human rating and agreement, the class Kaggle submission, Revanth's independent results, student review and the combined team report remain outstanding.

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

Srinidhi's task folders additionally contain a configuration and README. Task 3 also contains `evaluate_local.py`, `full_metrics_report.csv`, and `outputs/pred_A2B/` / `outputs/pred_B2A/`, as shown in the PDF's GAN tree. `submission.csv` is a generated deliverable; its schema and actual class outputs must be verified before creating it. The final team PDF will be `report/DATA266_Lab1_Report_Team_49.pdf` after the remaining team evidence and comparisons are complete.

Lab Pair 49 uses `srinidhi/` for [Mrnidhi](https://github.com/Mrnidhi) and `revanth0211/` for [Revanth0211](https://github.com/Revanth0211). Revanth's folders are empty contribution scaffolds; his independent implementation and results are pending. Shared runner/setup/test utilities remain in `src/`, `scripts/`, and `tests/`; execution receipts are in `verification/`. See [team workflow](TEAM_WORKFLOW.md).

## Windows setup and one-command smoke test

Use Python 3.12 and an isolated `.venv` from the repository root. The desktop run uses Python 3.12.14, the exact direct dependency versions in `requirements.txt`, and PyTorch `2.11.0+cu128` / torchvision `0.26.0+cu128`. Its CUDA 12.8 build supports the RTX 5090's `sm_120` architecture; actual hardware and installed packages are recorded in the [desktop environment receipt](verification/part1_desktop_environment.json) and the raw run's `provenance/` manifest.

```powershell
git clone https://github.com/Mrnidhi/Data266_Lab.git
cd Data266_Lab
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
```

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

## Proposed models

| Task | Srinidhi's proposed full model | Configuration |
|---|---|---|
| Character GPT | Manual causal attention, 3 blocks, 6 heads, width 192, context 256, 12 epochs | `task1_llm/srinidhi/config.json` |
| Yelp sentiment | Max-pool embedding MLP, BiLSTM, residual dilated CNN; learned embeddings | `task2_sentiment/srinidhi/config.json` |
| CycleGAN | Two 9-block generators and two PatchGAN discriminators, 256px, 30 epochs | `task3_gan/srinidhi/config.json` |

`smoke` uses tiny synthetic inputs and smaller models. `rehearsal` uses proposed dimensions with small public text subsets and short runs; GAN inputs remain explicitly synthetic until class data are supplied. `full` selects complete training. Full GAN mode refuses missing data/manifests. These configurations are starting hypotheses, not measured optima.

On an already allocated Linux NVIDIA machine:

```bash
bash scripts/bootstrap_gpu.sh
.venv/bin/python scripts/prefetch_rehearsal.py
.venv/bin/python scripts/rehearsal.py --device cuda --mode rehearsal --max-minutes 60
```

See each task's member README for full training, resume and evaluation. The [compute allocation plan](COMPUTE_PLAN.md) records the desktop placement, earlier cloud costs and checkpoint transfers. The [RunPod handoff](RUNPOD_HANDOFF.md) records the earlier optional cloud workflow; the new Part 1 run uses the desktop. The scripts do not allocate or stop a pod; their time limit stops training, not billing.

For visible training progress, see the [SSH terminal workflow](SSH_WORKFLOW.md). All model runners print flushed step/loss/elapsed/estimated-time updates; the connection launcher requires the actual running Pod's SSH host/port and a registered key.

## Data, evidence and submission

Raw text datasets are cached under each task's `data/huggingface/`; Srinidhi's selected rows, tokenizer mappings and preprocessing stay in his own `data_processed/`. Verified GAN images belong in `task3_gan/data/monet_jpg/` and `photo_jpg/`, with individual split manifests under the member's `data_processed/`. Data-folder READMEs are tracked so the structure is visible on GitHub; large data, environments, caches, ZIPs and most model binaries are ignored. The selected desktop Part 1 `best.pt` is published on the pushed review branch. Part 2 publication targets branch `srinidhi/part2-desktop-final`; the three selected member `checkpoints/<model>/best.pt` files have targeted Git-ignore exceptions. Check the final handoff and `git ls-remote` before relying on that remote branch to restore them. Both best/last weights and processed data are included in their verified local Part 1/Part 2 ZIPs. Save other weights separately and document their accessible location/checksum; include all required weights in the final Canvas ZIP. Never commit credentials or personal absolute paths.

Commit raw logs/manifests without editing them after a run. For the desktop Part 1 run, `metrics.jsonl` is the untouched step/epoch evidence tracked in Git and included in the ZIP. The original console `RUN_LOG.txt` remains local; startup library warnings contain machine-specific paths, so this console file is excluded from Git and the ZIP. The October 1 Part 1 replacement was explicitly requested by the user; it does not establish that the earlier missing checkpoints were recovered. New evidence uses portable relative paths. Each member must complete `results.md`, `failure_analysis.md` and `metrics_report.csv` using actual full-run evidence. Agree on the evaluation protocol before comparing models.

Finish `report/DATA266_Lab1_Report_Team_49.pdf`, include this repository link, compare all members and package the final Canvas submission as one ZIP containing separate Part 1, Part 2 and Part 3 folders plus the combined `Report.pdf`. Each part needs an executed notebook with visible outputs, actual model weights and applicable outputs. Individual Part 1/Part 2 archives do not complete the whole submission. Student manual error review/viva preparation, Revanth's independent results, team comparisons, the combined report and the remaining Part 3 requirements remain pending. See [submission checklist](SUBMISSION_CHECKLIST.md), [report outline](report/REPORT_TEMPLATE.md), [research notes](RESEARCH_NOTES.md), and [AI assistance disclosure](AI_USE.md).
