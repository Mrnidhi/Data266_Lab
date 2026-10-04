# Part 2: Yelp sentiment — Revanth

Three binary sentiment classifiers trained from scratch on the same Yelp
Polarity split: a mean-embedding baseline, a multi-kernel CNN, and a
bidirectional GRU. Each model learns its own embeddings; no pretrained model
or tokenizer is used. The completed run used seed 3143 and an NVIDIA GeForce
RTX 4090.

## Results and model comparison

The multi-kernel CNN produced the strongest test accuracy and macro-F1.

| Model | Selected epoch | Validation macro-F1 | Test accuracy | Test macro-F1 | Parameters |
|---|---:|---:|---:|---:|---:|
| Mean-embedding baseline | 5 | 0.926288 | 92.845% | 0.928449 | 5,120,129 |
| Multi-kernel CNN | 5 | **0.932577** | **93.775%** | **0.937748** | 6,707,969 |
| Bidirectional GRU | 4 | 0.929981 | 93.130% | 0.931299 | 6,622,977 |

The CNN improved test macro-F1 by about 0.93 percentage points over the
baseline. A paired McNemar test found that difference statistically
distinguishable on this test set (p = 1.25e-08). The BiGRU comparison with the
baseline was not significant at the 0.05 level (p = 0.0973).

The CNN also had the lowest Brier score, 0.04742, while the baseline had the
lowest 15-bin expected calibration error, 0.01034. These measures therefore
favor different models. The CNN took 23.65 seconds to train versus 9.19
seconds for the baseline and 80.69 seconds for the BiGRU in the recorded run.

## Models and data

| Model | Architecture | Learning rate / dropout |
|---|---|---:|
| Baseline | 128-dimensional embedding → masked mean pool → binary logit | 0.001 / 0.20 |
| CNN | 160-dimensional embedding → parallel width-3/5/7 convolutions, 128 channels each → global max pool → binary logit | 0.0007 / 0.30 |
| BiGRU | 160-dimensional embedding → bidirectional GRU, 128 units per direction → final states → binary logit | 0.0005 / 0.30 |

All models use a training-only vocabulary capped at 40,000 tokens, inputs
truncated or padded to 256 tokens, batch size 256, binary cross-entropy with
logits, and a maximum of five epochs. The data contain 100,000 training,
10,000 validation, and 20,000 test reviews. The notebook shuffles with the
fixed seed before selecting each split.

The raw dataset is intentionally not committed. The notebook downloads
`fancyzhx/yelp_polarity` through Hugging Face Datasets. The saved vocabulary
is [outputs/vocab.json](outputs/vocab.json), and preprocessing is documented
in the executed notebook.

## Robustness and error review

The evaluation includes accuracy; macro, micro, and weighted precision,
recall, and F1; ROC-AUC; PR-AUC; MCC; Brier score; ECE; confusion counts; and
500-resample bootstrap intervals. Paired McNemar tests compare both
experimental models with the baseline on the same test examples.

Long reviews were the CNN's weakest length slice, with error rate 0.06703.
Across every reported model and slice, the BiGRU long-review slice was worst
at 0.07716. The 256-token limit is important here because later conclusions
can be truncated before the model sees them.

[The 20-case review](outputs/error_review_20.csv) contains five confident
false positives, five confident false negatives, five near-threshold errors,
and five long/negation slice failures. Every row has a case-specific written
observation. The main patterns are mixed sentiment, negation and idioms,
sarcasm, borderline calibration, and missing context in long reviews.

## Read and reproduce

Start with:

- [Executed notebook](src/Part2.ipynb)
- [Detailed results](results.md)
- [Metrics table](metrics_report.csv)
- [Failure-analysis summary](failure_analysis.md)
- [Final configuration](config.json)
- [Output inventory](outputs/README.md)

The successful environment reported PyTorch 2.14.0+cu130 and CUDA 13.0.
Execute the notebook from its `src/` directory so its output paths resolve
correctly:

```bash
cd task2_sentiment/revanth0211/src
jupyter nbconvert --to notebook --execute Part2.ipynb \
  --output Part2_executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

A full execution downloads data, trains all three models, and rewrites the
member checkpoints and outputs. Preserve the published artifacts before
starting a new run.

## Evidence and limitations

The notebook retains visible outputs for preprocessing, model definitions,
training histories, full evaluation, plots, significance tests, robustness
slices, the completed error review, hardware disclosure, and artifact checks.
[Checkpoint hashes](checkpoints/manifest.json) and the preserved
[run log](RUN_LOG.txt) link the write-up to the saved files. The root
`metrics_report.csv` combines the original model metrics with the saved McNemar
results and slice metrics so the PDF's required metrics are in one file. Original
notebook outputs and raw logs remain unchanged; the derivation is recorded in
[the metric manifest](../../reproducibility/manifests/revanth0211/part2-metrics.json).

These measurements represent one seed and one selected split. Bootstrap
intervals quantify test-row sampling uncertainty, not variation across
training seeds. McNemar p-values are unadjusted pairwise tests. The official
test set is not a new hidden deployment domain, and the results do not
establish production readiness.

The code and write-up received AI assistance and were reviewed against the
saved run artifacts. Team comparisons, the combined report, and viva
preparation remain separate course requirements.
