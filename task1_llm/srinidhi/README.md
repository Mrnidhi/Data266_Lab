# Part 1 — Character GPT, Srinidhi

A character GPT trained from scratch on 100,000 TinyStories training stories and
10,000 validation stories. The selected run completed 16 epochs and 28,416 updates
on October 1, 2026. No pretrained model or tokenizer is used.

## Results

| Evaluation | Validation CE | Perplexity | Character accuracy |
|---|---:|---:|---:|
| Original model, context 256 | 0.721101 | — | 77.31% |
| Selected model, context 256 | 0.599380 | — | 80.91% |
| Selected model, trained context 512 | 0.551181 | 1.735301 | 82.44% |

Selection required at least 0.005 lower validation CE on identical 256-character
windows. At that context, dropout-disabled train CE is 0.591177 and the validation
minus training gap is +0.008204. Epoch 16 had the best native validation loss;
there was no late rise in the recorded curve. This does not establish consistently
coherent stories or performance on other domains. Bits per character at context
512 is 0.795186. Training steps took 111.49 minutes, with 117.86 minutes total;
shared GPU workload limits comparisons with isolated training runs.

## Model and data

Six decoder blocks use eight attention heads, width 256, feedforward width 1024,
context 512 and dropout 0.1: **4,928,512 parameters**. Attention explicitly computes
Q/K/V, scaled scores, causal masking, softmax and weighted values. No prebuilt
attention or Transformer block is used. See [gpt.py](src/gpt.py).

Training uses seed 2342, batch 128, BF16, AdamW at 0.0004, betas (0.9, 0.95), weight
decay 0.01 and gradient clipping 1.0. Learning rate warms up over 5% of updates,
then follows cosine decay to 10% of its peak. Depth, context and duration changed
together, so the comparison does not isolate the effect of any single change.

Stories are split before windowing, and vocabulary is fitted on training only.
Windows stay within stories; shifted targets score each character/EOS once per
epoch. Padding is ignored; four unseen validation characters map to UNK.

## Setup and reproduction

Work from the repository root or the extracted ZIP's `Part 1/` folder. Use Python
3.12; the recorded GPU environment was PyTorch 2.11.0+cu128 / CUDA 12.8.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m lab1.run --task gpt --mode smoke --device cpu
```

On Linux/macOS use `python3.12` and `.venv/bin/python`. Select that environment as
the notebook kernel. Smoke uses synthetic data and checks execution only. NVIDIA
training needs a compatible driver. For a fresh full run:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode full --device cuda --config task1_llm/srinidhi/outputs/full/reproduction_config.json --set offline=true --output runs/part1-new-full
```

The ZIP includes the frozen cache. A clone without it needs `--set offline=false`
to prepare public TinyStories. The member `config.json` is the original baseline
recipe; use the explicit selected recipe above. Hardware/library changes can
change numerical results. Full mode covers every story window each epoch.

Resume an interrupted matching run using its original output, config and overrides,
adding `--resume <matching-run>/checkpoints/last.pt`. This restores optimizer,
scheduler, AMP, RNG and progress state. Never overwrite a completed run.

## Results notebook and packaging

Open [gpt.ipynb](src/gpt.ipynb) for preprocessing, manual attention, metrics, curves,
ten continuations and saved-model inference. It does not retrain. The 38-row
[metric report](metrics_report.csv) and [failure analysis](failure_analysis.md)
record measurement limits and three real generation failures.

To verify an existing completed run and rebuild its portable archive:

```powershell
.venv\Scripts\python.exe scripts/finalize_gpt.py --run-dir reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part1/depth_context_full
.venv\Scripts\python.exe scripts/finalize_gpt.py --verify-archive dist/Part1_Srinidhi_2342.zip
Get-FileHash -Algorithm SHA256 dist/Part1_Srinidhi_2342.zip
```

Finalization checks epoch coverage, metrics, source/data/checkpoint hashes,
notebook execution and fresh saved-model inference without training. Receipts are
in `verification/part1_*.json`; ZIP inventory, CRC and hashes are in
`dist/part1_package_verification.json`. An extracted notebook/CPU smoke check is
recorded separately from a clean dependency installation. The package contains
best/last weights and data. Comparing against the original baseline also needs
its separately preserved checkpoint. Original machine-specific console logs stay
local; portable `metrics.jsonl` retains the training record.

The three failure interpretations received AI assistance and still need student
review. See [requirements](REQUIREMENTS_CHECKLIST.md),
[contributions and assistance](../../README.md#contributions-and-assistance) and
[submission](../../README.md#submission). This individual package does not replace
the team comparison, combined report or viva preparation.
