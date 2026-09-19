# Part 2 research and validation search

Research reviewed September 18, 2026. These are hypotheses to test, not guaranteed improvements or claims to match a paper's scores. All embeddings stay randomly initialized and trained on the frozen Yelp training split.

## Evidence and transfer limits

- [Johnson and Zhang, DPCNN (2017)](https://aclanthology.org/P17-1052.pdf) studies deeper word-level CNNs for document classification, including Yelp Polarity. Its pyramid reductions, validation protocol and optimizer differ from ours; variants with unsupervised embeddings are outside this lab's scratch-only comparison. The useful idea here is testing a larger context window, not borrowing a reported accuracy.
- [Bai et al., TCN (2018)](https://arxiv.org/abs/1803.01271) motivates residual dilated convolutions as alternatives to recurrence. It does not establish a Yelp score for our noncausal classifier.
- [Zhang and Wallace (2017)](https://aclanthology.org/I17-1026.pdf) investigates CNN width, receptive regions, regularization and variability. It mainly studies shallow sentence models with pretrained embeddings, so its guidance is only a reason for controlled width/dropout trials here.
- [Thakar et al., Yelp study (2026)](https://aclanthology.org/2026.acl-srw.38.pdf) uses about 21,000 reviews and three sentiment classes. Its best pretrained RoBERTa system is not permitted by this lab and is not directly comparable to our full binary Yelp Polarity experiment. Recency alone does not make a method a valid match.

## Why these trials

The frozen validation data lose only about 0.93% of reviews to the 384-token cap, and mean per-review OOV is about 0.95%. Vocabulary expansion and a longer input cap are therefore lower priority. The current CNN has a 61-token receptive field (two kernel-3 convolutions for each dilations 1, 2, 4 and 8). Adding dilations 16 and 32 expands this to 253 tokens, nearer the 95th-percentile review length of 232.

The three challengers are a longer-context CNN, a moderately wider BiLSTM, and a wider max-pool MLP. Exact settings live in experiments/*.json. Each family also has a control with the original architecture and the same 12-epoch maximum and patience of 4. A longer patience gives the reduced learning rate time to work; the original patience of 2 could stop at the first LR reduction.

All runs use the same frozen training/validation split, vocabulary and 384-token cap. Candidate training does not receive a test dataset. The reference suite already has normal final test evaluations, which are preserved but excluded from tuning decisions. Compare validation macro-F1, prefer the smaller model within 0.001, and treat that margin as a practical rule, not a significance test. Record every trial, including failures. Confirm a small winning gain with another initialization on the same split when needed; do not use a different data split as a seed confirmation.

After selection is frozen, evaluate selected checkpoints on all 38,000 test rows and recompute paired comparisons. Report both the reference and selection process. There is no claim that a finite search finds a global optimum.
