# Part 1: Character GPT — Srinidhi

The replacement full run completed on October 1, 2026 on the Windows desktop's
NVIDIA GeForce RTX 5090. It trained from random initialization on 100,000
TinyStories training stories and 10,000 validation stories for 12 complete
epochs and 18,732 updates. The user requested deletion of the previous Part 1 runs after their
saved weights could not be located. The new run is the sole source of this
Part 1 publication; no previous numerical results are carried forward.

Raw evidence: `reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full/`.
The selected epoch-12 checkpoint has validation cross-entropy **0.721101**,
perplexity **2.056697**, bits per character **1.040329** and next-character
accuracy **77.31%**. Training cross-entropy at that epoch is **0.798289**.
The run trained in BF16 with 1,427,904 parameters, averaging 1.186 million
scored character/EOS targets per second during timed training work. Timed
training steps took 15.11 minutes; recorded end-to-end run time was 18.32 minutes.
These metrics do not imply consistently coherent generated stories.
The [requirement mapping](REQUIREMENTS_CHECKLIST.md)
separates technical evidence from student review and remaining team obligations.

## Architecture and training recipe

The model uses 3 pre-LayerNorm Transformer blocks, 6 attention heads, embedding
width 192, feedforward width 768, context length 256 characters and dropout
0.15. Token and positional embeddings are learned. Attention explicitly
calculates QK-transpose, scaling, the causal mask, softmax and multiplication
by V. No pretrained model/tokenizer, prebuilt attention module or Transformer
block is used. The source is `src/gpt.py` in this member folder.

The full configuration in `config.json` uses seed 2342, batch size 256, AdamW
at learning rate 0.0004, betas (0.9, 0.95), weight decay 0.01, gradient clipping
at 1.0, 5% linear warm-up and cosine decay to 10% of the peak rate. It runs 12
complete epochs without a step cap. These are the existing configured design
choices, not evidence of optimality or new batch-size comparisons on this GPU.

The teammate's proposed scaffold differs in architecture and hyperparameters;
his independently trained work and comparative results remain pending. This
repository's implementation and explanations received AI assistance. The
student must verify the design and interpretations and be able to defend them;
see [the assistance disclosure](../../AI_USE.md).

## Reproduce on Windows

Follow the root README to create `.venv` and install the pinned dependencies
and editable package. The verified desktop environment uses Python 3.12.14,
PyTorch 2.11.0+cu128, CUDA 12.8 and an RTX 5090. The environment receipt is
`verification/part1_desktop_environment.json` at the repository root.

One-command offline CPU smoke check after setup:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode smoke --device cpu
```

The exact command used for this fresh desktop full run was:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode full --device cuda --output reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full
```

For another full reproduction, choose a new empty `--output` directory. The
runner refuses to overwrite an existing run. Full mode needs the public
TinyStories dataset download or the verified frozen cache. No cloud rental is
needed on this CUDA-equipped desktop.

To resume this same full run from its last saved checkpoint:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode full --device cuda --output reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full --resume reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full/checkpoints/last.pt
```

Keep the complete run directory with both best and last checkpoints when
resuming. On Linux/macOS use `.venv/bin/python` in the equivalent commands.

## Data preparation and modes

- `smoke`: Small synthetic stories and a smaller network; an offline CPU check.
- `rehearsal`: 512 official training stories and 64 validation stories, the
  configured full architecture and at most 64 updates. Its seeded streaming
  buffer is not a uniform sample of the entire dataset or final quality evidence.
- `full`: 100,000 distinct official training stories and 10,000 distinct official
  validation stories selected with seeded indexed permutations. Source revision,
  row IDs and SHA-256 text hashes are frozen in the manifest. Full training
  traverses every fixed-length window each epoch.

`lab1.gpt.prepare_data(config)` returns `(training_texts, validation_texts,
manifest)` and writes local JSONL caches without training.
`lab1.gpt.data_cache_paths(config)` reports the exact cache paths. Full caches
are member-specific under `data_processed/full/`; raw downloads are shared
under `task1_llm/data/huggingface/`. Copy the complete verified cache for offline
training. With `offline=true`, a missing or corrupt cache causes an error.

The train-only character vocabulary contains PAD, UNK, BOS and EOS plus sorted
training characters. The source constructs its own encoding/decoding dictionaries;
the notebook explicitly displays `char_to_idx` and `idx_to_char` constructed
from the saved token order. Unknown validation
characters map to UNK and their count is reported. Splits are made before
window construction; windows never cross story boundaries. Each story's
next-character/EOS targets occur exactly once per full epoch, including the
final padded window. Padding does not contribute to loss or accuracy. The
epoch target-count assertion checks complete coverage.

## Metrics and saved evidence

The run writes resolved configuration, source/data manifests, vocabulary,
untouched `metrics.jsonl` step/epoch records, training/validation loss curves,
training diagnostics, fixed-prompt greedy and sampled generations, summaries,
and `checkpoints/best.pt` plus `checkpoints/last.pt`. The original `RUN_LOG.txt`
console remains local and is excluded from Git because startup library warnings
include machine-specific paths; its training records are not rewritten.

`metrics_report.csv` reports all required metrics: train/validation
cross-entropy, perplexity, bits per character, next-character accuracy,
generalization gap, Distinct-1/2/3, repeated 4-gram rate, gradient norms and
stability, parameter count, train/generation tokens per second, peak memory
and training time. Character metrics include EOS targets; perplexity from a
different tokenizer is not directly comparable. The loss gap is validation
eval-mode loss minus online training-mode loss. Diversity uses continuation-only
lowercased regex word tokens; empty n-gram denominators produce null. Training
throughput excludes loading, validation and checkpoint I/O; wall-clock runtime
is reported separately.

Checkpoint files contain model, optimizer, scheduler, AMP scaler, RNG states,
split manifest, vocabulary and within-epoch progress. A deterministic batch
ordering supports resume at the next unfinished batch. The run reloads a fresh
model and compares inference logits on the same device; finalization also
verifies saved weights and executes the results notebook. Exact floating-point
agreement across different GPU/PyTorch environments is not guaranteed.

The selected `checkpoints/best.pt` has a targeted Git-ignore exception for
publication; once committed and pushed, a clone supplies actual trained Part 1
weights. The resume checkpoint `last.pt` and processed story cache
are retained locally and included in the Part 1 submission ZIP. Large raw
downloads and the ZIP remain ignored by Git. See [checkpoint details](checkpoints/README.md)
for hashes and the package verification record.

## Student and team work remaining

Read all actual continuations and verify the three evidence-based failure
analysis drafts. Review the notebook and explain causal attention, padding,
warm-up, checkpoint selection and each metric for the viva. AI-generated prose
does not establish the student's own understanding or rubric compliance.
Revanth's independent results, the joint comparisons and combined report remain
separate deliverables. The final Canvas ZIP must also include Parts 2 and 3 and
one combined `Report.pdf` with the GitHub link.

## References

- [Official TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [TinyStories paper](https://arxiv.org/abs/2305.07759)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
