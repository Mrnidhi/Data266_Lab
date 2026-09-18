# Part 2 — Yelp sentiment, Srinidhi

Three independent classifiers learn their embeddings from scratch. The main comparison holds the data, preprocessing, vocabulary, length cap, batch size, checkpoint selection, and evaluation protocol constant.

| Model | Full/rehearsal architecture | Initial learning rate |
| --- | --- | --- |
| `maxpool_mlp` | Embedding128 → masked max pooling → Dense64/ReLU/dropout → binary logit | 0.0015 |
| `bilstm` | Embedding128 → one-layer bidirectional LSTM,96 units per direction → masked max pooling → Dense64/ReLU/dropout → logit | 0.001 |
| `dilated_cnn` | Embedding128 →128 channels → four residual blocks with two kernel3 convolutions each, dilation1/2/4/8 → masked max pooling → Dense64/ReLU/dropout → logit | 0.0008 |

Full settings: seed2342, vocabulary cap40,000, minimum frequency2, maximum384 tokens, batch128, AdamW with weight decay0.0001, gradient clipping1.0, at most6 epochs, early stopping after2 epochs without validation macro-F1 improvement. Dropout is0.3 in embedding/head locations and0.2 within CNN blocks. Learning rate halves after a validation-loss plateau. Each model starts independently; no pretrained embedding or model is downloaded.

The residual CNN masks padding after **each convolution**. The BiLSTM packs sequence lengths and masks its pooling. These details prevent longer padding within a batch from changing a review's prediction.

## Profiles

- `smoke`:24/12/16 synthetic train/validation/test examples; one epoch; smaller widths. Checks execution and artifacts, not Yelp accuracy.
- `rehearsal`:2,048/256/512 official Yelp rows; full proposed architectures and384-token cap; two epochs. Intended to measure resource use and catch GPU problems. Not final results.
- `full`: official560,000 training reviews split90/10 into504,000 training and56,000 validation rows; all38,000 official test rows remain untouched. Same seeded stratified split for all three models.

The vocabulary is fitted only on training rows. Preprocessing expands contractions, removes HTML/punctuation and a frozen customized stopword list, and preserves negation/contrast terms such as `not`, `never`, and `but`. Unknown or empty reviews map to `<unk>`. Duplicate normalized raw text is audited; official test rows are never silently deleted. Duplicate counts and truncation/OOV rates are saved.

## Local commands

Run from the project root with its prepared virtual environment:

```bash
.venv/bin/python -m lab1.run --task sentiment --mode smoke --device cpu --output runs/sentiment-smoke
```

To cache official rehearsal data before renting any GPU:

```python
import json
from pathlib import Path
from lab1.sentiment import prefetch_data
config = json.loads(Path("task2_sentiment/srinidhi/config.json").read_text())["rehearsal"]
prefetch_data(config, Path("task2_sentiment/srinidhi/data_processed/rehearsal"))
```

This explicit function downloads the official dataset via Hugging Face, selects the frozen rows, and writes only the selected JSONL splits plus checksums. It does not start training. A prepared cache can be copied to another machine and passed through `data_dir`:

```bash
.venv/bin/python -m lab1.run --task sentiment --mode rehearsal --device cuda --set 'data_dir="task2_sentiment/srinidhi/data_processed/rehearsal"' --output runs/sentiment-rehearsal
```

For final training, prefetch the `full` profile separately or leave its `data_dir` unset for a normal official-dataset download. Rehearsal data cannot be used as full data; counts, seed, mode, and file hashes are checked.

## Checkpoints and resume

Every model writes `checkpoints/best.pt` and `checkpoints/last.pt`, containing model/optimizer/scheduler/scaler state, epoch history, vocabulary, configuration, Python/NumPy/PyTorch/CUDA RNG states, and the shuffle-generator state. Resume works at **completed epoch boundaries**; an interrupted partial epoch repeats. Model checkpoints are local trusted artifacts; do not load arbitrary third-party pickle checkpoints.

```bash
.venv/bin/python -m lab1.run --task sentiment --mode rehearsal --device cuda --set 'data_dir="task2_sentiment/srinidhi/data_processed/rehearsal"' --output runs/sentiment-rehearsal --resume runs/sentiment-rehearsal
```

Use the same profile/configuration and data. Local cache paths may change when moving machines. The resume fingerprint verifies configuration, vocabulary, IDs, labels, and review content. A completed model is evaluated from its best checkpoint; unfinished models continue. A run never overwrites existing training checkpoints unless `resume` is supplied.

## Saved evaluation

Each model exports test-row IDs, true labels, predictions and positive probabilities; per-class and macro/micro/weighted precision, recall and F1; accuracy; confusion matrix; ROC-AUC; **trapezoidal PR-AUC**; MCC; Brier; and top-label confidence ECE using15 equal-width bins. Curves, training loss, validation macro-F1, parameter count, epoch timing, examples/second, and peak CUDA allocated memory are also saved.

The full profile uses1,000 paired-index IID bootstrap draws for95% percentile intervals on accuracy, macro-F1, and MCC. These intervals cover test-sample variation, not training-seed variation. Exact two-sided paired McNemar tests compare each experiment against the baseline on identical test rows; the two prespecified p-values are reported unadjusted. Slice files include support and metrics for review length, negation, and OOV rate. Single-class slices have undefined ROC/PR-AUC recorded as null.

`error_candidates_for_manual_review.csv` supplies up to20 confidently incorrect examples **per model**. The human fields remain blank. Review them and assign categories such as negation, sarcasm, mixed sentiment, truncation/missed context, or ambiguous labels; write a short evidence-based explanation. If fewer than20 errors exist, the file reports the available count rather than inventing examples. The run summary correctly keeps `manual_error_review_complete: false` until a person completes the report work.

## Validation and evidence boundary

Run `.venv/bin/python -m pytest tests/test_sentiment.py -q`. Tests check padding invariance across all three models, preservation of negation, vocabulary isolation, known-answer metrics, paired-test construction, independent model shapes, split overlap rejection, and synthetic training/checkpoint reload/resume. A successful CPU smoke test does not establish CUDA speed, memory fit, full training completion, model quality, or a completed manual error analysis.

Architecture reference: [Bai et al., temporal convolutional networks](https://arxiv.org/abs/1803.01271). Dataset: [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity). The dilated CNN here is a small classification adaptation; no published performance claim is assumed.
