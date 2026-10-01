# Part 1 — desktop run and portable package

Repository: https://github.com/Mrnidhi/Data266_Lab

This package contains Srinidhi's replacement character GPT trained from scratch
on the Windows desktop RTX 5090 on October 1, 2026. The completed run uses
100,000 TinyStories training stories, 10,000 validation stories, 12 complete
epochs and 18,732 updates. The selected epoch-12 checkpoint achieved validation
cross-entropy **0.7211**, character perplexity **2.0567** and next-character
accuracy **77.31%**. The raw run is
`reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full/`.
Training completion and all selected metrics are established by its `summary.json`,
the published `task1_llm/srinidhi/results.md` and verification receipts.

The user requested replacement of previous Part 1 runs after the older saved
weights could not be located. This package's metrics and checkpoints come
only from the fresh desktop run. No cloud rental is needed to reproduce it on
an appropriately configured NVIDIA desktop.

## Contents and scope

`dist/Part1_Srinidhi_2342.zip` at the repository root contains one `Part 1/`
folder. Extract that folder and open it as the working directory. Paths inside
the archive retain the repository layout:

- `task1_llm/srinidhi/src/gpt.ipynb`: executed notebook with visible outputs.
- `task1_llm/srinidhi/checkpoints/best.pt` and `last.pt`: selected and resumable
  checkpoints; `manifest.json` supplies file sizes and SHA-256 hashes.
- `task1_llm/srinidhi/data_processed/full/`: frozen processed training and
  validation stories plus their split manifest. Character vocabulary token
  order is saved in the raw run and published `outputs/full/vocabulary.json`;
  the notebook displays both integer mapping dictionaries.
- `task1_llm/srinidhi/outputs/full/`: curves, diagnostics, generations and
  vocabulary/data metadata; `metrics_report.csv`, `results.md` and
  `failure_analysis.md` are in the member folder.
- `reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full/`: portable
  untouched step/epoch metrics, history, source/data manifests and run summaries.
- `verification/`: desktop environment, saved-checkpoint inference, notebook
  and export/finalization records.
- `src/`, `scripts/`, `pyproject.toml` and `requirements.txt`: runtime,
  reproducible setup and finalization tools.

The original console `RUN_LOG.txt` is retained locally without editing it.
Library startup warnings contain machine-specific paths, so it is omitted
from Git and the ZIP. The untouched `metrics.jsonl` is the portable raw
training evidence. The selected Part 1 `best.pt` has a targeted Git-ignore
exception for publication; its presence in a remote commit must be checked
before claiming a push or relying on a clone to restore it. The larger
processed story cache and resume `last.pt` travel in the ZIP.

This is a Part 1 technical package. Student review of the AI-assisted analysis,
independent teammate results, the team comparison and combined report remain
separate requirements. Canvas asks for one final ZIP with separate Part 1,
Part 2 and Part 3 folders and one combined `Report.pdf` with the GitHub link.
Do not submit this Part 1 ZIP as though it were the entire lab.

## Windows setup

From the cloned repository root or extracted `Part 1/` root, use Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
```

The repository's requirements cover all three tasks. The standalone archive
uses generated Part 1-only requirements and package metadata, so its setup
does not need GAN evaluation extras. The desktop's recorded environment is
Python 3.12.14, PyTorch 2.11.0+cu128 and CUDA 12.8; its RTX 5090 `sm_120`
support was checked before training. The exact installed versions are in
`verification/part1_desktop_environment.json` and the raw provenance manifest.
Select `.venv\Scripts\python.exe` as the VS Code notebook kernel. If the
`py` launcher is unavailable, create `.venv` with an installed Python 3.12
executable instead.

After setup, reproduce an offline smoke check with **one command**:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode smoke --device cpu
```

This uses synthetic stories and a smaller model and writes to a new timestamped
folder under `runs/`. It establishes that the runner works; full-data results
come from the complete recorded run.

On Linux/macOS create the environment with `python3.12 -m venv .venv` and
replace `.venv\Scripts\python.exe` with `.venv/bin/python` in the commands.
GPU full reproduction requires CUDA-capable PyTorch and compatible NVIDIA
drivers. The smoke command works on CPU.

## Training, resume and finalization

The original desktop command was:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode full --device cuda --output reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full
```

Do not rerun it into a populated directory. For a new full reproduction use
a fresh output path, for example:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode full --device cuda --set offline=true --output runs/part1-new-full
```

Offline full training uses the included hash-verified frozen story cache.
For a clone without the cache, omit `--set offline=true` to prepare the public
TinyStories data. The split source revision, row identifiers and text hashes
remain explicit in each new manifest. Full mode does not accept a step cap
as a completed run.

To resume the original run, keep its complete evidence directory and use:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode full --device cuda --output reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full --resume reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full/checkpoints/last.pt
```

The checkpoint carries model, optimizer, scheduler, AMP, RNG, vocabulary,
manifest and within-epoch progress. Floating-point results can differ across
GPU/PyTorch environments even with the same split and seed.

After training completes, this command verifies the run, publishes Part 1,
executes only its results notebook, checks saved inference and builds the ZIP:

```powershell
.venv\Scripts\python.exe scripts/finalize_gpt.py --run-dir reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full
```

The export-only command is:

```powershell
.venv\Scripts\python.exe scripts/publish_gpt_results.py --run-dir reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full
```

The notebook displays the actual run's metrics, curves and generations and
verifies saved checkpoint inference; executing it does not repeat twelve
training epochs. The finalizer cross-checks complete epoch/window coverage,
selected checkpoint, metric identities, untouched raw logs and processed-data
hashes. See `verification/part1_export.json`, `part1_notebook.json`,
`part1_checkpoint_inference.json` and `part1_finalization.json` for recorded
checks. The ZIP CRC and SHA-256 receipt is
`dist/part1_package_verification.json` in the producing repository.

Before transferring the ZIP, verify its checksum against that receipt and
retain a second copy outside the workstation. From the producing repository:

```powershell
Get-FileHash -Algorithm SHA256 dist/Part1_Srinidhi_2342.zip
.venv\Scripts\python.exe scripts/finalize_gpt.py --verify-archive dist/Part1_Srinidhi_2342.zip
```

Read `task1_llm/srinidhi/REQUIREMENTS_CHECKLIST.md` for the rubric mapping and
the student/team obligations that automated checks cannot establish.
