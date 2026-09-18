# DATA266 Lab 1 — team repository

Shared repository: https://github.com/Mrnidhi/Data266_Lab

Each member independently implements and trains all three tasks. Members keep code, configurations, preprocessing, outputs and write-ups in their own named folder under each task. The team agrees on an evaluation protocol and submits one combined report.

## Current status

Srinidhi's implementation is prepared and locally tested. Committed notebook outputs are labeled **CPU smoke checks**, not final training results. Other members should add their own named folders and independently implement their models. GPU rehearsal and full training are pending. The PDF's class Kaggle link currently returns “We can't find that page,” so actual class GAN data and submission rules still need verification.

## Layout

```text
task1_llm/srinidhi/       # Character GPT: source, notebook, config, results
task2_sentiment/srinidhi/ # Three Yelp classifiers
task3_gan/srinidhi/       # CycleGAN training/evaluation/inference
src/lab1/               # Runner and provenance utilities
scripts/                # Setup, data preparation and verification
tests/                  # Correctness and resume tests
reproducibility/         # Unedited run logs and manifests, grouped by member
report/                 # Combined team report (pending)
```

Only Srinidhi's member folders currently exist. Teammates should use the same structure with their own names. Shared tooling does not replace independent model implementations. See [team workflow](TEAM_WORKFLOW.md).

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

See each task's member README for full training, resume and evaluation. See [RunPod handoff](RUNPOD_HANDOFF.md) for the proposed paid rehearsal. The scripts do not allocate or stop a pod; their time limit stops training, not billing.

## Data, evidence and submission

Datasets, environments, caches, preparation ZIPs and model binaries are ignored by Git. Prepare data with the scripts or supply verified class data. Save large weights outside ordinary Git and document an accessible download location and checksum in each member's `checkpoints/README.md`; the final Canvas ZIP must contain required weights. Never commit credentials, API keys or personal absolute paths.

Commit raw logs/manifests without editing them after a run. Preserve original local evidence for earlier checks; new committed checks are generated with portable relative paths. Each member must complete `results.md`, `failure_analysis.md` and `metrics_report.csv` using actual full-run evidence. Agree on the evaluation protocol before comparing models.

Finish `report/DATA266_Lab1_Report_Team_<number>.pdf`, include this repository link, compare all members and package the final Canvas submission. See [submission checklist](SUBMISSION_CHECKLIST.md), [report outline](report/REPORT_TEMPLATE.md), [research notes](RESEARCH_NOTES.md), and [AI assistance disclosure](AI_USE.md).
