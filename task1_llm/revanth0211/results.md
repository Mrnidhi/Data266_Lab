# Part 1 results

## Run summary

I trained a character-level GPT-style language model from scratch using manually implemented causal multi-head self-attention. The final model has 25,385,984 parameters, eight attention heads, eight transformer blocks, a model width of 512, and a context length of 256 characters. Training used 100,000 sequences, with another 10,000 sequences held out for validation.

The target validation accuracy was reached at epoch 13. The run completed without NaN values and took 1,431.13 seconds (about 23.9 minutes) on an NVIDIA GeForce RTX 4090.

## Measured results

| Metric | Result |
|---|---:|
| Best epoch | 13 |
| Validation cross-entropy | 0.622993 |
| Validation perplexity | 1.864500 |
| Bits per character | 0.898789 |
| Top-1 next-character accuracy | 0.805136 |
| Train-validation loss gap | 0.084886 |
| Repeated 4-gram rate | 0.291262 |
| Training throughput | 232,543 tokens/second |
| Peak GPU memory | 3,687.76 MB |

Validation accuracy improved from 0.684755 after the first epoch to 0.8051 at epoch 13. The validation loss generally decreased throughout training, although the small increase at epochs 10 and 12 shows that the improvement was not perfectly smooth.

## Interpretation

The model learned the spelling, punctuation, and general tone of a simple children's story. It could produce locally readable phrases, but the generated sample still repeated ideas, combined words in grammatically incorrect ways, and ended with an incomplete sentence. The three concrete examples and proposed tests are documented in `failure_analysis.md`.

The next experiment I would run is a decoding comparison using the same checkpoint: the current settings versus a lower temperature, a no-repeat 3-gram rule, and sentence-aware stopping. This would show whether the visible generation problems come mainly from decoding or require additional training.
