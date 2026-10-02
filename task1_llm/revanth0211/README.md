# Character language model — Revanth0211

This folder contains my completed DATA 266 Lab 1 Part 1 character-level GPT experiment. The model was implemented from scratch with manual causal multi-head self-attention; it does not use `nn.Transformer` or `nn.MultiheadAttention`.

The selected epoch-13 checkpoint reached 0.805136 held-out top-1 next-character accuracy, 0.622993 validation cross-entropy, and 1.864500 perplexity. Training used 100,000 TinyStories windows, 10,000 validation windows, and an NVIDIA GeForce RTX 4090.

## Contents

- `src/Part1.ipynb`: canonical executed notebook with visible outputs
- `config.json`: final training and generation settings
- `checkpoints/best_model.pt`: selected checkpoint
- `metrics_report.csv`: one-row results summary
- `outputs/`: metrics, history, training log, loss curve, and generated sample
- `results.md`: measured-result interpretation
- `failure_analysis.md`: three exact generated failure examples and testable mitigations

The raw dataset is intentionally not committed. The notebook downloads `roneneldan/TinyStories` through Hugging Face Datasets when local text files are unavailable.

Run the notebook from `task1_llm/revanth0211/src/` so its relative paths resolve to this member folder.
