# Part 2 results

## Run summary

I trained three sentiment classifiers from scratch on the same Yelp Polarity split: a mean-embedding baseline, a multi-kernel CNN, and a bidirectional GRU. Each model used a learned vocabulary of 40,000 tokens and a maximum input length of 256 tokens. The evaluation used 20,000 held-out test reviews.

## Model comparison

| Model | Best epoch | Accuracy | Macro-F1 | ROC-AUC | Brier score | ECE | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean-embedding baseline | 5 | 0.92845 | 0.92845 | 0.97830 | 0.05423 | **0.01034** | 9.19 s |
| Multi-kernel CNN | 5 | **0.93775** | **0.93775** | **0.98455** | **0.04742** | 0.01756 | 23.65 s |
| Bidirectional GRU | 4 | 0.93130 | 0.93130 | 0.98198 | 0.05158 | 0.01983 | 80.69 s |

The CNN was the strongest overall model. It improved accuracy and macro-F1 by about 0.93 percentage points over the baseline while taking about 2.6 times as long to train. Its local filters appear to capture useful sentiment phrases without the much larger runtime of the BiGRU.

The BiGRU improved slightly over the baseline, but it was the slowest model and did not perform best on the long-review slice. Its long-review macro-F1 was 0.92007, compared with 0.93084 for the CNN. A likely reason is that all models were limited to 256 tokens, so sequential modeling could not recover information that had already been truncated.

## Statistical and calibration findings

The paired McNemar comparison between the baseline and CNN produced a p-value of 1.25 × 10⁻⁸, so their difference was statistically distinguishable on this test set. The baseline-versus-BiGRU p-value was 0.0973, which was not significant at the 0.05 level.

The CNN had the lowest Brier score, meaning its probability estimates had the best overall squared error. The baseline had the lowest expected calibration error, so no single model won every calibration measure.

## Robustness and error analysis

Long reviews were the CNN's weakest length slice, with an error rate of 0.06703. Across all reported model-slice combinations, the BiGRU on long reviews had the highest error rate at 0.07716. Manual review also showed repeated problems with mixed sentiment, negation, idioms, sarcasm, and conclusions that fell beyond the 256-token input window.

The 20 reviewed examples are documented in `outputs/error_review_20.csv` and summarized in `failure_analysis.md`. A useful next experiment would compare the current 256-token input with a 512-token setting while keeping the train/validation/test split fixed. I would also add targeted hard examples containing sentiment reversals, idioms, and negation.
