# Part 1: Character GPT — Srinidhi

Full training completed on an RTX 4090: 12 epochs on 100,000 training and 10,000
validation stories. The selected checkpoint is epoch 12, with validation loss
0.7197, character perplexity 2.0538 and next-character accuracy 77.38%.
See `results.md`, `metrics_report.csv`, `outputs/full/` and the executed notebook.
These metrics do not imply that the generated stories are consistently coherent.

The full candidate uses 3 pre-LayerNorm Transformer blocks, 6 heads, width 192,
feedforward width 768, context 256 characters, dropout 0.15, batch size 256,
AdamW at 0.0004, 5% warm-up and cosine decay, and 12 complete epochs. The seed is
2342. Unlike the teammate's 4-layer, 4-head, width-256, context-128 model, this
tests a narrower model with more character context. These choices are not a
claim of optimality; validate them against the actual compute budget.

An RTX 4090 rehearsal on September 18 measured approximately 207K, 612K,
701K, and 755K character targets/second at batch sizes 32, 128, 256, and 512.
Batch 256 was selected for throughput with about 4 GB peak tensor memory;
512 gave only another 8% while doubling memory and halving updates per epoch.
These short rehearsals establish speed, not final quality. The learning rate
and 12 full epochs remain unchanged. The initial batch-32 full-run attempt
was deliberately interrupted for this tuning and is retained as an incomplete
experiment, separate from the new batch-256 run.

The attention implementation explicitly calculates QK-transpose, scaling,
causal masking, softmax, and multiplication by V. Positions are learned. There
is no pretrained tokenizer/model, prebuilt attention layer, or Transformer block.

## Three modes

- `smoke`: Tiny synthetic stories and a smaller network; CPU correctness checks
  without any download. This is not a quality or full-architecture benchmark.
- `rehearsal`: 512 real official training stories and 64 validation stories,
  the complete proposed architecture, at most 64 updates. A seeded streaming
  buffer avoids downloading the entire training dataset; this is not a uniform
  sample of all TinyStories. The result is explicitly marked as rehearsal.
- `full`: 100,000 distinct official training stories and 10,000 distinct official
  validation stories, independent seeded permutations, and 12 complete passes.
  No step cap is accepted in full mode. Dataset revision, row IDs and SHA-256
  text hashes are recorded. The resolved revision is fixed at preparation time.

The shared runner selects the complete mode configuration from `task1_llm/srinidhi/config.json`.
The module interface is `lab1.gpt.run(config, output_dir, device, resume=None)`.
Use the project runner instructions at the repository root for command lines.

## Preparing data before GPU time

`lab1.gpt.prepare_data(config)` returns `(training_texts, validation_texts,
manifest)` and creates local JSONL caches without constructing or training a
model. `lab1.gpt.data_cache_paths(config)` returns their paths. The configured
cache directory defaults to `.cache/tinystories`, relative to the working directory.
Copy the entire cache folder to the training environment. Verified cache hits
need no network. Set `offline=true` to fail instead of downloading missing data.
Do this before the training run. Resume preserves the model, schedule and data contract;
cache paths, offline/loading settings and checkpoint/log frequency may change between machines.

The train-only character vocabulary includes PAD, UNK, BOS and EOS. Unknown
validation characters map to UNK, and counts are reported. Stories are split
before window construction; no window crosses a story boundary. Each story's
next-character and EOS targets occur exactly once in a full epoch, including
the final padded window. Padding is excluded from loss and accuracy. The epoch
target-count assertion checks that no windows were silently dropped.

## Saved evidence

The output directory contains the complete configuration, data manifest,
vocabulary, JSONL step/epoch metrics, learning curves, diagnostic curves,
fixed-prompt greedy/sampled generations, a summary, and `checkpoints/best.pt`
and `checkpoints/last.pt`. Metrics cover cross-entropy, character perplexity,
bits per character, next-character accuracy, train/validation loss gap,
Distinct-1/2/3, repeated 4-grams, parameter count, gradient norms, speed,
memory and elapsed time. Character metrics include EOS targets; comparison
with another tokenizer's perplexity is invalid. The reported gap is validation
eval-mode loss minus online training-mode loss, rather than a matched evaluation
of both splits. Diversity uses continuation-only lowercased regex word tokens.
Empty n-gram denominators produce null, not a fabricated zero or perfect score.

Checkpoint files contain model, optimizer, scheduler, AMP scaler, random states,
split manifest, vocabulary and within-epoch progress. A deterministic batch
ordering lets a same-environment resume continue at the next unfinished batch.
Use checkpoint files generated by this project. Keep both best and last files
together when moving a run, so the prior best remains available after resuming.
The run verifies a checkpoint by reloading a fresh model and comparing inference
logits on the same device. Identical results across different GPU/PyTorch versions
are not guaranteed. Moving from CPU/BF16 to FP16 initializes a fresh gradient
scaler when no saved FP16 scaler exists; this is recorded in the run summary.

## Student work still required

Review the actual generations and the three evidence-based, AI-assisted drafts in
`failure_analysis.md`. Check their interpretations and explain the attention,
masking, training and metric code for the viva. The drafts identify repetition,
broken grammar and character identity drift; they do not claim student review.
The raw training logs and all ten generated continuations are preserved.

## Sources informing the design

- [Official TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [TinyStories paper](https://arxiv.org/abs/2305.07759)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Perplexity and evaluation context](https://huggingface.co/docs/transformers/main/perplexity)
- [2026 data/context scaling study](https://arxiv.org/html/2602.07488v3): its
  larger BPE models motivate testing context; their numerical settings are not
  evidence that this character model's settings are optimal.
- [Regional-TinyStories, December 2025](https://aclanthology.org/2025.findings-ijcnlp.142/):
  separate grammar/fluency from story consistency/completeness in human review.
