# DATA266 Lab 1 — Lab Pair 49

Shared repository: https://github.com/Mrnidhi/Data266_Lab

Each member independently implements and trains all three tasks. Members keep code, configurations, preprocessing, outputs and write-ups in their own named folder under each task. The team agrees on an evaluation protocol and submits one combined report.

## Current status

Srinidhi's **Part 1 full training completed from scratch on the Windows desktop RTX 5090** on October 1, 2026: 100,000 training stories, 10,000 validation stories, 12 complete epochs and 18,732 updates. The selected epoch-12 checkpoint achieved validation cross-entropy **0.7211**, character perplexity **2.0567** and next-character accuracy **77.31%**. The user requested removal and replacement of the previous Part 1 runs; these numbers come only from the new desktop run. Its untouched evidence lives in `reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full/`. The [results](task1_llm/srinidhi/results.md), [38-row metric report](task1_llm/srinidhi/metrics_report.csv), [notebook](task1_llm/srinidhi/src/gpt.ipynb) and actual selected weights are published in the member folder. The [finalization guide](PART1_FINALIZATION.md) documents setup, inference/notebook checks and the portable `dist/Part1_Srinidhi_2342.zip`; [the requirement mapping](task1_llm/srinidhi/REQUIREMENTS_CHECKLIST.md) identifies the remaining student/team obligations.

Part B full training and all six validation-only comparison runs completed on one RTX 5090. All 96 cloud artifacts (1.37 GB) were verified locally before the pod was stopped at $0.00/hour. Final CPU evaluation on all 38,000 test reviews achieved 96.01% accuracy for BiLSTM, 95.80% for CNN and 93.20% for the MLP. The [executed notebook](task2_sentiment/srinidhi/src/sentiment.ipynb), [results](task2_sentiment/srinidhi/results.md) and [20-case AI-assisted review draft](task2_sentiment/srinidhi/failure_analysis.md) are available. Candidate recipes, research rationale and the frozen selection rule are in [Part B research and search](task2_sentiment/srinidhi/RESEARCH_AND_SEARCH.md).

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

Raw text datasets are cached under each task's `data/huggingface/`; Srinidhi's selected rows, tokenizer mappings and preprocessing stay in his own `data_processed/`. Verified GAN images belong in `task3_gan/data/monet_jpg/` and `photo_jpg/`, with individual split manifests under the member's `data_processed/`. Data-folder READMEs are tracked so the required structure is visible on GitHub; large data, environments, caches, ZIPs and most model binaries are ignored. The selected desktop Part 1 `task1_llm/srinidhi/checkpoints/best.pt` has a targeted Git-ignore exception for publication; after it is committed and pushed, a clone will include real trained weights. The resumable `last.pt` and processed story cache are retained in the local Part 1 ZIP. Save other weights separately and document their accessible location/checksum in each `checkpoints/README.md`; include required weights in the final Canvas ZIP. Never commit credentials or personal absolute paths.

Commit raw logs/manifests without editing them after a run. For the desktop Part 1 run, `metrics.jsonl` is the untouched step/epoch evidence tracked in Git and included in the ZIP. The original console `RUN_LOG.txt` remains local; startup library warnings contain machine-specific paths, so this console file is excluded from Git and the ZIP. The October 1 Part 1 replacement was explicitly requested by the user; it does not establish that the earlier missing checkpoints were recovered. New evidence uses portable relative paths. Each member must complete `results.md`, `failure_analysis.md` and `metrics_report.csv` using actual full-run evidence. Agree on the evaluation protocol before comparing models.

Finish `report/DATA266_Lab1_Report_Team_49.pdf`, include this repository link, compare all members and package the final Canvas submission as one ZIP containing separate Part 1, Part 2 and Part 3 folders plus the combined `Report.pdf`. Each part needs an executed notebook with visible outputs, actual model weights and applicable outputs. A Part 1 archive alone is not the full Canvas submission. Student review/viva preparation, Revanth's independent results, team comparisons, the combined report and the remaining Part 3 requirements remain pending. See [submission checklist](SUBMISSION_CHECKLIST.md), [report outline](report/REPORT_TEMPLATE.md), [research notes](RESEARCH_NOTES.md), and [AI assistance disclosure](AI_USE.md).
