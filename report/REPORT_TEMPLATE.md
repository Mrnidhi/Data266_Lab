# Lab Pair 49 — Lab 1 report outline

This outline has no final results. Replace every pending field using actual outputs and your own explanation. Export the finished combined report to `DATA266_Lab1_Report_Team_49.pdf` with the GitHub repository link.

## Reproducibility

Team/member names; repository URL and commit; SID4/seed; data sources/revisions and split counts; preprocessing; exact hardware/software; run folder IDs and checkpoint hashes. State which experiments ran on cloud versus college hardware, and disclose AI assistance as the course requires.

## Part 1: character-level generation

Explain manual causal attention, the architecture, character vocabulary, next-character labels, training recipe and selection rule. Include train/validation curves; cross-entropy, perplexity, BPC, generalization gap and accuracy; generated continuations; distinct n-grams and repeated 4-grams; gradient/stability traces; parameters, throughput, memory and time. Explain three actual failures and what experiment might address each. Compare with all teammates under a consistent evaluation setup.

## Part 2: sentiment classification

Explain the three architectures and why their comparisons are informative. State stopword/negation handling, vocabulary fitting, padding/truncation, official test preservation and validation-based selection. Include each model's accuracy; precision/recall/F1 averages; confusion matrix; ROC-AUC and PR-AUC; MCC; Brier/ECE and reliability plot; required bootstrap intervals; paired McNemar tests; and slice results. Discuss 20 selected errors using the requested categories. Add parameters, hardware, memory and timings, plus teammate comparisons.

## Part 3: unpaired translation

Explain both generators/discriminators, unpaired domain sampling, exact loss coefficients, image preprocessing and selection rule. Include both-direction loss curves and fixed grids. Report FID/KID and sample counts, generative precision/recall, cycle L1, LPIPS, content-feature cosine, and stability/compute metrics with method names and limitations. Include 30 fixed samples rated independently by two actual humans, the scores and agreement. Document own-checkpoint inference and actual Kaggle submission details/score once available. Discuss content/style/artifact failures and teammate comparisons.

## Conclusions

What the actual comparisons support, uncertainty and remaining failure cases. Cite the course materials, dataset sources, research and any borrowed implementation ideas. Do not use synthetic checks or rehearsal estimates as final findings.
