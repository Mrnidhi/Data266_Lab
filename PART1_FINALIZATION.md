# Part 1: selected character GPT and portable package

Repository: https://github.com/Mrnidhi/Data266_Lab

Srinidhi's selected model trained from scratch on 100,000 training and 10,000
validation TinyStories for **16 complete epochs and 28,416 updates** on the
Windows desktop RTX 5090. Epoch 16 had the best validation loss. At its trained
context length of 512, CE is **0.551181**, perplexity **1.735301**, bits per
character **0.795186** and next-character accuracy **82.44%**.

The model was selected using the rule frozen before training. On identical
256-character windows, validation CE fell from baseline **0.721101** to
**0.599380**, exceeding the required 0.005 improvement. With dropout disabled,
train CE is **0.591177** and the validation-minus-train gap is **+0.008204**.
Validation loss improved throughout training; there was no late rise in the
recorded curve. These results do not prove reliable story coherence or broad
production performance.

The selected source is
`reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part1/depth_context_full/`.
The original baseline remains in its raw run, metadata snapshot and
`dist/Part1_Baseline_20261001.zip`. Original source/configuration and raw evidence
are preserved. The selected recipe is in
`task1_llm/srinidhi/outputs/full/reproduction_config.json`; the original member
`config.json` remains the baseline recipe.

## What to open

Extract `dist/Part1_Srinidhi_2342.zip` and open its `Part 1/` folder as the working
directory. It contains the member's executed notebook, best/last weights, frozen
story cache, outputs, metric CSV, failure analysis, source and setup files.
The notebook explains preprocessing and manual attention, shows all required
metrics/curves and ten actual continuations, and checks saved-model inference.
It does not train the model again.

The comparison receipt is included, but repeating the two-model comparison
requires the separately preserved baseline run and checkpoint. History copies
and local backup checkpoints are excluded from the selected ZIP. The original
console log stays local because library warnings include machine-specific
paths; unedited `metrics.jsonl` provides the portable training log.

## Setup and one smoke command

From the repository or extracted `Part 1/` root, use Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m lab1.run --task gpt --mode smoke --device cpu
```

Select `.venv\Scripts\python.exe` as the notebook kernel. If `py` is unavailable,
use an installed Python 3.12 interpreter. On Linux/macOS use `python3.12` and
`.venv/bin/python`. CPU smoke uses synthetic data and a smaller model; it is
an execution check, not full training evidence. The recorded desktop uses
Python 3.12.14, PyTorch 2.11.0+cu128 and CUDA 12.8. Full GPU training needs a
compatible NVIDIA device/driver; the standalone requirements omit GAN extras.

## Reproduce or resume training

Use the selected recipe and a new empty output directory:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode full --device cuda --config task1_llm/srinidhi/outputs/full/reproduction_config.json --set offline=true --output runs/part1-new-full
```

The ZIP contains the verified story cache. A Git clone does not include that
cache; use `--set offline=false` to prepare public TinyStories data. Full mode
covers every story window in every epoch and does not treat a step-capped run
as complete. Numerical results can differ across GPU/PyTorch environments.

To resume a genuinely interrupted matching run, retain its full evidence
folder and checkpoint, use its original output path and add:

```text
--resume <matching-run>/checkpoints/last.pt
```

Use the same explicit reproduction config and original overrides. The
checkpoint contains model, optimizer, scheduler, AMP, RNG and progress state.
Do not restart completed training into its existing evidence directory.

## Verify existing results and build the ZIP

After training and the grounded three-case failure analysis are complete:

```powershell
.venv\Scripts\python.exe scripts/finalize_gpt.py --run-dir reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part1/depth_context_full
```

This verifies full epoch/target coverage, source/data/checkpoint hashes and
metric identities, publishes selected outputs, executes the seven-cell results
notebook, checks fresh CPU/CUDA inference and creates the Part 1 ZIP. It does
not start another training run. The three failure excerpts are checked against
the actual saved text; their AI-assisted interpretations still need student review.

Receipts are in `verification/part1_export.json`, `part1_notebook.json`,
`part1_checkpoint_inference.json`, `part1_finalization.json` and
`part1_package_portability.json`. The portability check executes the notebook
and CPU smoke from a fresh extraction with the pinned existing interpreter;
it is not a clean dependency installation. The ZIP inventory, CRC and SHA-256
receipt is `dist/part1_package_verification.json`.

```powershell
Get-FileHash -Algorithm SHA256 dist/Part1_Srinidhi_2342.zip
.venv\Scripts\python.exe scripts/finalize_gpt.py --verify-archive dist/Part1_Srinidhi_2342.zip
```

Keep a verified backup outside the workstation. Best/last weights are real
checkpoint bytes in the ZIP; selected `best.pt` is prepared for normal Git
publication to `main`. Check the remote commit before relying on a clone.

## Submission scope

This is Srinidhi's Part 1 package. Student understanding/viva, independent
teammate results, team comparisons and the combined report remain required.
See `task1_llm/srinidhi/REQUIREMENTS_CHECKLIST.md` and `AI_USE.md`. Canvas needs
one ZIP with separate Part 1, Part 2 and Part 3 folders and a combined
`Report.pdf` containing the repository link. This individual ZIP does not
complete the whole assignment.
