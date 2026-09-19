# Part 2 research and validation search

Research reviewed September 18, 2026. These are hypotheses to test, not guaranteed improvements or claims to match a paper's scores. All embeddings stay randomly initialized and trained on the frozen Yelp training split.

## Evidence and transfer limits

- [Johnson and Zhang, DPCNN (2017)](https://aclanthology.org/P17-1052.pdf) studies deeper word-level CNNs for document classification, including Yelp Polarity. Its pyramid reductions, validation protocol and optimizer differ from ours; variants with unsupervised embeddings are outside this lab's scratch-only comparison. The useful idea here is testing a larger context window, not borrowing a reported accuracy.
- [Bai et al., TCN (2018)](https://arxiv.org/abs/1803.01271) motivates residual dilated convolutions as alternatives to recurrence. It does not establish a Yelp score for our noncausal classifier.
- [Zhang and Wallace (2017)](https://aclanthology.org/I17-1026.pdf) investigates CNN width, receptive regions, regularization and variability. It mainly studies shallow sentence models with pretrained embeddings, so its guidance is only a reason for controlled width/dropout trials here.
- [Thakar et al., Yelp study (2026)](https://aclanthology.org/2026.acl-srw.38.pdf) uses about 21,000 reviews and three sentiment classes. Its best pretrained RoBERTa system is not permitted by this lab and is not directly comparable to our full binary Yelp Polarity experiment. Recency alone does not make a method a valid match.

## Why these trials

The 384-token cap truncates about 0.93% of the frozen validation reviews, and mean per-review OOV is about 0.95%. Vocabulary expansion and a longer input cap are therefore lower priority. The current CNN has a 61-token receptive field (two kernel-3 convolutions for each dilations 1, 2, 4 and 8). Adding dilations 16 and 32 expands this to 253 tokens, nearer the 95th-percentile review length of 232.

The three challengers are a longer-context CNN, a moderately wider BiLSTM, and a wider max-pool MLP. Exact settings live in experiments/*.json. Each family also has a control with the original architecture and the same 12-epoch maximum and patience of 4. A longer patience gives the reduced learning rate time to work; the original patience of 2 could stop at the first LR reduction.

All runs use the same frozen training/validation split, vocabulary and 384-token cap. Candidate training does not receive a test dataset. The reference suite already has normal final test evaluations, which are preserved but excluded from tuning decisions. Compare validation macro-F1, prefer the smaller model within 0.001, and treat that margin as a practical rule, not a significance test. Record every trial, including failures. Confirm a small winning gain with another initialization on the same split when needed; do not use a different data split as a seed confirmation.

After selection is frozen, evaluate selected checkpoints on all 38,000 test rows and recompute paired comparisons. Report both the reference and selection process. There is no claim that a finite search finds a global optimum.

## Completed validation comparison

All nine candidates finished. The following table was frozen before the selected-model CPU test evaluation. No new trial was started after test inspection.

| Candidate | Validation macro-F1 | Selected epoch | Parameters | Selected |
| --- | ---: | ---: | ---: | --- |
| reference:maxpool_mlp | 0.926570 | 5 | 5,128,321 | Yes |
| reference:bilstm | 0.956607 | 4 | 5,305,985 | No |
| reference:dilated_cnn | 0.953892 | 6 | 5,539,073 | Yes |
| maxpool_mlp_long_schedule_control | 0.926570 | 5 | 5,128,321 | No |
| bilstm_long_schedule_control | 0.958802 | 10 | 5,305,985 | Yes |
| dilated_cnn_long_schedule_control | 0.953892 | 6 | 5,539,073 | No |
| mlp_wider | 0.926959 | 3 | 10,273,025 | No |
| bilstm_wider | 0.958661 | 8 | 8,026,241 | No |
| cnn_long_context | 0.953714 | 6 | 5,736,193 | No |

The original-width BiLSTM benefited from the longer schedule and selected epoch 10. The larger BiLSTM did not beat it. The wider MLP gained less than the prespecified 0.001 tolerance, so the smaller reference was retained. The larger CNN receptive field did not improve validation macro-F1. Both selected MLP and CNN remained the reference checkpoints. These are observations from one initialization and one frozen split, not evidence of an overall optimum. The configured optional seed-confirmation condition was not triggered.
