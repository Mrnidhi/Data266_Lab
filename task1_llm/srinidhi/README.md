# Part 1: character GPT — Srinidhi

A character GPT trained from scratch on 100,000 TinyStories training stories
and 10,000 validation stories. The selected run completed 16 epochs and
28,416 updates on the desktop RTX 5090 on October 1, 2026. No pretrained model
or tokenizer is used.

## Results and model choice

The deeper model passed the rule fixed before training: reduce validation
cross-entropy by at least 0.005 using the same 256-character evaluation windows.
Both models used the same frozen stories and character vocabulary.

| Same evaluation setting | Original model | Selected model |
|---|---:|---:|
| Validation cross-entropy | 0.721101 | 0.599380 |
| Next-character accuracy | 77.31% | 80.91% |

At its trained context length of 512, the selected model has validation CE
**0.551181**, perplexity **1.735301**, bits per character **0.795186** and
next-character accuracy **82.44%**. These context-512 scores are reported
separately from the fair comparison above. The baseline and its ZIP remain
preserved; the comparison receipt is `verification/part1_quality_comparison.json`.

For an overfitting check, both splits were evaluated with dropout disabled
using 256-character windows: train CE **0.591177**, validation CE **0.599380**,
gap **+0.008204**. Native validation loss improved at every recorded epoch;
epoch 16 was the best. This shows no late validation deterioration in this run,
but does not establish performance on new domains or consistently coherent stories.

## How the model works

Characters become learned token and position embeddings. Six decoder blocks
use eight attention heads, width 256, feedforward width 1024 and dropout 0.1.
Each block uses LayerNorm, residual connections and a feedforward network.
Attention explicitly computes Q/K/V, scaled scores, a causal mask, softmax
and the weighted values. No prebuilt attention or Transformer block is used.
There are **4,928,512 parameters**; the implementation is in `src/gpt.py`.

Training uses seed 2342, batch 128, AdamW at 0.0004, betas (0.9, 0.95), weight
decay 0.01 and gradient clipping at 1.0. The learning rate warms up over 5%
of updates, then follows cosine decay to 10% of its starting peak. Training
uses BF16. Dropout and weight decay reduce overfitting; validation loss selects
the saved best checkpoint. Changing depth, context and duration together does
not show which single change caused the improvement.

The story split is frozen before windowing. Vocabulary is fitted only on
training text. Windows stay within stories, shift targets by one character,
and score every character/EOS target once per epoch. Padding is ignored;
four unseen validation characters map to UNK. The notebook displays the
character-to-index and index-to-character dictionaries.

## Run it

Install the pinned environment using [the setup guide](../../PART1_FINALIZATION.md).
From the repository or extracted Part 1 root:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode smoke --device cpu
```

For a new full reproduction, use the selected recipe and a fresh output:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode full --device cuda --config task1_llm/srinidhi/outputs/full/reproduction_config.json --set offline=true --output runs/part1-new-full
```

The archive includes the frozen cache. For a clone without that cache, use
`--set offline=false` to prepare the public data. `config.json` retains the
original baseline recipe; the explicit reproduction config above matches
the selected model. See [the finalization guide](../../PART1_FINALIZATION.md)
for resume, notebook and package verification commands.

## Evidence and remaining work

`src/gpt.ipynb` displays preprocessing, manual attention, all required metrics,
loss curves, raw training output and actual greedy/sampled text. The 38-row
`metrics_report.csv` includes loss, perplexity, accuracy, generalization gap,
diversity/repetition, stability, parameters, throughput, memory and time.
The required training gap compares online training loss with validation loss;
the same-mode overfitting check above is a separate, fairer comparison.

Training steps took 111.49 minutes; recorded total run time was 117.86 minutes.
Part of training shared the GPU with sentiment work, so these are measured
run costs, not an isolated speed comparison. Generations still repeat phrases
or lose meaning; [failure analysis](failure_analysis.md) gives three exact cases.

The source run is
`reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part1/depth_context_full/`.
Saved best/last weights and their hashes are in `checkpoints/`; raw evidence
remains unchanged. The verified Part 1 ZIP contains both weights and the data.

The code and explanations received AI assistance. Student review and viva
understanding, the teammate's independent model, team comparisons and the
combined report remain separate requirements. See [the mapping](REQUIREMENTS_CHECKLIST.md)
and [AI disclosure](../../AI_USE.md). A Part 1 ZIP is only one part of the
final three-part Canvas submission.
