# Part 1: character GPT — Revanth

A character-level GPT trained from scratch on TinyStories. The model uses
manual causal multi-head self-attention and does not call
`nn.Transformer` or `nn.MultiheadAttention`. The selected run used
100,000 training windows, 10,000 validation windows, seed 3143, and an
NVIDIA GeForce RTX 4090.

## Results

Training stopped at epoch 13 after reaching the validation-accuracy target.
The selected accuracy checkpoint was also the lowest-loss checkpoint from
this run.

| Metric | Measured value |
|---|---:|
| Validation cross-entropy | **0.622993** |
| Validation perplexity | **1.864500** |
| Bits per character | **0.898789** |
| Top-1 next-character accuracy | **80.5136%** |
| Online train-to-validation CE gap | +0.084886 |
| Parameters | 25,385,984 |
| Training throughput | 232,543 scored characters/second |
| Generation throughput | 301.60 characters/second |
| Peak allocated GPU memory | 3,687.76 MB |
| Recorded training time | 1,431.13 seconds |

Validation accuracy rose from 68.48% to 80.51%. Validation loss generally
fell across the run, with small temporary increases at epochs 10 and 12.
The reported gap compares online training-mode loss with evaluation-mode
validation loss, so dropout and within-epoch model updates affect that number.
It is evidence of a remaining split gap, not a perfectly controlled
same-mode overfitting estimate.

## Model and training design

The vocabulary contains 100 characters collected from both the training and
validation story pools. Validation characters therefore informed the vocabulary;
this is a limitation of the recorded preprocessing.
Characters are mapped to learned token and position embeddings. Eight
pre-normalized decoder blocks use eight attention heads, width 512, a
four-times-wider GELU feed-forward layer, residual connections, and dropout
0.1. Input and language-model head weights are tied. The context length is
256 characters.

Training uses AdamW with learning rate 0.0003, betas (0.9, 0.95), weight decay
0.02, and gradient clipping at 1.0. The schedule warms up for 5% of planned
updates and then follows cosine decay to 8% of the starting peak. CUDA
autocast uses BF16 when supported. The selected checkpoint is based on
validation next-character accuracy, with validation loss breaking a tie.

TinyStories are split into disjoint training and validation story pools before
sampling. Each fixed-length example remains inside a single story, and start
positions are sampled without replacement from the valid pool. The notebook
downloads `roneneldan/TinyStories` through Hugging Face Datasets when local
text files are unavailable; raw data are intentionally not committed.

## Read and reproduce

Start with:

- [Executed notebook](src/Part1.ipynb)
- [Detailed results](results.md)
- [Metrics table](metrics_report.csv)
- [Failure analysis](failure_analysis.md)
- [Final configuration](config.json)

The successful environment reported PyTorch 2.14.0+cu130 and CUDA 13.0.
Open or execute the notebook from this member folder or its `src/`
directory so relative paths resolve correctly:

```bash
cd task1_llm/revanth0211/src
jupyter nbconvert --to notebook --execute Part1.ipynb \
  --output Part1_executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

A full execution downloads data and rewrites the member checkpoint and output
artifacts. Preserve the published files before starting a new run.

## Evidence and limitations

The canonical notebook retains its executed outputs, including environment,
data, vocabulary, architecture, training, metrics, plots, generation, and
artifact checks. [The outputs guide](outputs/README.md) maps the saved
evidence. [The checkpoint manifest](checkpoints/manifest.json) records the
selected checkpoint hash and size.

The generated sample resembles a children's story locally but still repeats
phrases, loses grammatical coherence, and ends mid-sentence. Three exact
examples and testable decoding changes are documented in
[failure_analysis.md](failure_analysis.md). Diversity scores describe one
500-character generation and should not be treated as a broad quality study.
The run represents one seed, one split, and one dataset domain.

The code and write-up received AI assistance and were reviewed against the
saved run artifacts. Team comparisons, the combined report, and viva
preparation remain separate course requirements.
