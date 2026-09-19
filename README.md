# DATA266 Lab 1 — Lab Pair 49

Shared repository: https://github.com/Mrnidhi/Data266_Lab

Each member independently implements and trains all three tasks. Members keep code, configurations, preprocessing, outputs and write-ups in their own named folder under each task. The team agrees on an evaluation protocol and submits one combined report.

## Current status

Srinidhi's **Part A full training completed on an RTX 4090**: 12 epochs, 100,000 training stories and 10,000 validation stories. The selected epoch-12 checkpoint has validation cross-entropy 0.7197, character perplexity 2.0538 and next-character accuracy 77.38%. See the [executed notebook](task1_llm/srinidhi/src/gpt.ipynb), [results](task1_llm/srinidhi/results.md), [metrics](task1_llm/srinidhi/metrics_report.csv), and [observed failure cases](task1_llm/srinidhi/failure_analysis.md). GPU reload and fresh local CPU inference passed. The pod was stopped after a verified local checkpoint backup.

Part B full training and a bounded validation-only comparison are running on one RTX 5090. Candidate recipes, research rationale and the frozen selection rule are in [Part B research and search](task2_sentiment/srinidhi/RESEARCH_AND_SEARCH.md). The pod will stop after Part B evaluation and verified local backups. Part C training is deferred; the class archive has now been supplied. Revanth's independent implementation and results, student review and the combined team report remain outstanding.

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

Srinidhi's task folders additionally contain a configuration and README. Task 3 also contains `evaluate_local.py`, `full_metrics_report.csv`, and `outputs/pred_A2B/` / `outputs/pred_B2A/`, as shown in the PDF's GAN tree. `submission.csv` is a generated deliverable; its schema and actual class outputs must be verified before creating it. The final team PDF will be `report/DATA266_Lab1_Report_Team_49.pdf` after real experiments.

Lab Pair 49 uses `srinidhi/` for [Mrnidhi](https://github.com/Mrnidhi) and `revanth0211/` for [Revanth0211](https://github.com/Revanth0211). Revanth's folders are empty contribution scaffolds; his independent implementation and results are pending. Shared runner/setup/test utilities remain in `src/`, `scripts/`, and `tests/`; execution receipts are in `verification/`. See [team workflow](TEAM_WORKFLOW.md).

## Reproduce a smoke test with one command

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

See each task's member README for full training, resume and evaluation. The [compute allocation plan](COMPUTE_PLAN.md) prioritizes college/free GPUs, explains parallel task placement, costs and checkpoint transfers. The [RunPod handoff](RUNPOD_HANDOFF.md) is an optional paid fallback. The scripts do not allocate or stop a pod; their time limit stops training, not billing.

For visible training progress, see the [SSH terminal workflow](SSH_WORKFLOW.md). All model runners print flushed step/loss/elapsed/estimated-time updates; the connection launcher requires the actual running Pod's SSH host/port and a registered key.

## Data, evidence and submission

Raw text datasets are cached under each task's `data/huggingface/`; Srinidhi's selected rehearsal rows and preprocessing stay in his own `data_processed/`. Verified GAN images belong in `task3_gan/data/monet_jpg/` and `photo_jpg/`, with individual split manifests under the member's `data_processed/`. Data-folder READMEs are tracked so the required structure is visible on GitHub; large data, environments, caches, ZIPs and model binaries are ignored. Save weights separately and document their accessible location/checksum in `checkpoints/README.md`; include required weights in the final Canvas ZIP. Never commit credentials or personal absolute paths.

Commit raw logs/manifests without editing them after a run. Preserve original local evidence for earlier checks; new committed checks are generated with portable relative paths. Each member must complete `results.md`, `failure_analysis.md` and `metrics_report.csv` using actual full-run evidence. Agree on the evaluation protocol before comparing models.

Finish `report/DATA266_Lab1_Report_Team_49.pdf`, include this repository link, compare all members and package the final Canvas submission. See [submission checklist](SUBMISSION_CHECKLIST.md), [report outline](report/REPORT_TEMPLATE.md), [research notes](RESEARCH_NOTES.md), and [AI assistance disclosure](AI_USE.md).
