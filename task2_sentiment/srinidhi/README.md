# Part 2 — Yelp sentiment, Srinidhi

Three independent classifiers learn their embeddings from scratch. The main comparison holds the data, preprocessing, vocabulary, length cap, batch size, checkpoint selection, and evaluation protocol constant.

The reference suite and six validation-only trials completed on one RTX 5090. Selection retained the reference MLP (epoch 5), the longer-schedule BiLSTM (epoch 10 of 12), and the reference CNN (epoch 6). All cloud files were hash-verified locally before stopping the pod; final evaluation uses the local Apple M3 CPU. See `outputs/full/selection_manifest.json` for all nine candidates and `results.md` for measured outcomes.

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
- `full`: official560,000 training reviews split90/10 into504,000 training and56,000 validation rows; all38,000 official test rows are reserved for final evaluation, never validation-based candidate selection. Same seeded stratified split for all three models.

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

The full dataset is now cached locally. Prepare exact encoded features on CPU before paid training:

```bash
.venv/bin/python scripts/prepare_sentiment_features.py --mode full \
  --data-dir task2_sentiment/srinidhi/data_processed/full \
  --output task2_sentiment/srinidhi/data_processed/full_encoded
```

Pass `--set 'encoded_cache="task2_sentiment/srinidhi/data_processed/full_encoded"'` together with the full `data_dir` override when training. The cache uses NumPy arrays without pickle and verifies file checksums, preprocessing source, vocabulary, complete row content, and configuration. Changing the model recipe requires rebuilding this cache. It preserves tokens, splits, and evaluation statistics exactly.

CUDA data loaders use pinned memory and asynchronous transfers. The BiLSTM computes sequence lengths on CPU before transfer, avoiding a GPU-to-CPU synchronization. `num_workers` can be tuned after benchmarking; its default remains zero. These changes preserve the training recipe. `scripts/benchmark_sentiment.py` measures each full-size model on real cached reviews, compares worker counts, and reports timing estimates separately from final model results.

## Checkpoints and resume

Every model writes `checkpoints/best.pt` and `checkpoints/last.pt`, containing model/optimizer/scheduler/scaler state, epoch history, vocabulary, configuration, Python/NumPy/PyTorch/CUDA RNG states, and the shuffle-generator state. New runs also save `last.pt` every **250 training batches** by default (`checkpoint_every_steps`), including the within-epoch position and loss/count totals. Resume reconstructs the batch order and continues after the saved batch. Only work since the last saved checkpoint repeats. Older checkpoints remain supported at their saved epoch boundaries. Model checkpoints are local trusted artifacts; do not load arbitrary third-party pickle checkpoints.

```bash
.venv/bin/python -m lab1.run --task sentiment --mode rehearsal --device cuda --set 'data_dir="task2_sentiment/srinidhi/data_processed/rehearsal"' --output runs/sentiment-rehearsal --resume runs/sentiment-rehearsal
```

Use the same profile/configuration and data. Local cache paths may change when moving machines. The resume fingerprint verifies configuration, vocabulary, IDs, labels, and review content. A completed model is evaluated from its best checkpoint; unfinished models continue. A run never overwrites existing training checkpoints unless `resume` is supplied.

Move the complete suite directory so each classifier keeps both its best and last checkpoint. CPU/BF16 checkpoints initialize a fresh FP16 gradient scaler when resumed on a device that needs one; the metric record reports this transition. Different GPU/PyTorch environments can produce numerical differences. See [portable run instructions](../../../COMPUTE_PLAN.md) for verified transfer bundles and independent storage.

## Saved evaluation

Each model exports test-row IDs, true labels, predictions and positive probabilities; per-class and macro/micro/weighted precision, recall and F1; accuracy; confusion matrix; ROC-AUC; **trapezoidal PR-AUC**; MCC; Brier; and top-label confidence ECE using15 equal-width bins. Curves, training loss, validation macro-F1, parameter count, epoch timing, examples/second, and peak CUDA allocated memory are also saved.

The full profile uses1,000 paired-index IID bootstrap draws for95% percentile intervals on accuracy, macro-F1, and MCC. These intervals cover test-sample variation, not training-seed variation. Exact two-sided paired McNemar tests compare each experiment against the baseline on identical test rows; the two prespecified p-values are reported unadjusted. Slice files include support and metrics for review length, negation, and OOV rate. Single-class slices have undefined ROC/PR-AUC recorded as null.

`error_candidates_for_manual_review.csv` supplies up to20 confidently incorrect examples **per model**. The published `required_20_errors_for_review.csv` selects the rubric-specific groups: five confident false positives, five confident false negatives, five near-threshold errors and five long-review slice errors. The human fields remain blank. Review them and assign categories such as negation, sarcasm, mixed sentiment, truncation/missed context, or ambiguous labels; write a short evidence-based explanation. If fewer than20 errors exist, the file reports the available count rather than inventing examples. The run summary correctly keeps `manual_error_review_complete: false` until a person completes the report work.

## Validation and evidence boundary

Run `.venv/bin/python -m pytest tests/test_sentiment.py -q`. Tests check padding invariance across all three models, preservation of negation, vocabulary isolation, known-answer metrics, paired-test construction, independent model shapes, split overlap rejection, and synthetic training/checkpoint reload/resume. A successful CPU smoke test does not establish CUDA speed, memory fit, full training completion, model quality, or a completed manual error analysis.

Architecture reference: [Bai et al., temporal convolutional networks](https://arxiv.org/abs/1803.01271). Dataset: [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity). The dilated CNN here is a small classification adaptation; no published performance claim is assumed.

## Validation-only comparison and result selection

The reference suite is preserved alongside six additional runs: the original architecture with a longer stopping schedule and one architecture/width challenger per model family. See [research rationale](RESEARCH_AND_SEARCH.md) and the exact `experiments/*.json` recipes. Candidate training receives only train and validation datasets. The reference suite already performed its normal test evaluation; those test scores are excluded from candidate decisions.

The tuning runner first verifies the encoded cache against the original feature configuration, then applies allowed training-only overrides. This safely reuses the same tokens and split while the generic full runner still enforces its complete cache fingerprint.

```bash
.venv/bin/python scripts/tune_sentiment.py \
  --candidate task2_sentiment/srinidhi/experiments/bilstm_wider.json \
  --output runs/part-b-tuning/bilstm_wider --device cuda

# After the reference and all six named comparison runs have completed:
.venv/bin/python scripts/select_sentiment_candidates.py --output runs/part-b-selection.json
.venv/bin/python scripts/finalize_sentiment_selection.py \
  --selection runs/part-b-selection.json --output runs/part-b-selected --device cpu
.venv/bin/python scripts/publish_sentiment_results.py \
  --run-dir runs/part-b-selected --data-dir task2_sentiment/srinidhi/data_processed/full
```

The selection file freezes validation scores, exact configurations, source provenance and checkpoint hashes before final evaluation. Within 0.001 macro-F1 of a family's best validation score, the smaller model is preferred. This practical tolerance is not a significance test. Final evaluation recomputes metrics and paired comparisons on all 38,000 test IDs. Published training times reflect overlapping jobs on a shared RTX 5090 and should not be treated as isolated speed benchmarks.

`scripts/part2_status.py` prints all run histories and live GPU load when invoked on the training host. Published notebook cells use the local `outputs/full` artifacts and selected checkpoint files; they do not require a running pod. Notebook generation and notebook execution are separate steps.
