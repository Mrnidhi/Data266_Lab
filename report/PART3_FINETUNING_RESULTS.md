# Part 3 CycleGAN continuation results

Date: 29 September 2026. The model comparison, test evaluation, direct exports, complete local backup verification, and GPU shutdown are complete. A separate local results package includes an executed notebook. Human ratings and final submission are still pending; no exact rental bill is claimed.

The validation rule selected the identity-weight-2.5 checkpoint at continuation update **44,992**, corresponding to additional epoch **8**. Its mean validation KID is **17.97% lower** than the original checkpoint's rechecked score. Its subsequently measured mean test KID is **7.52% lower** than the original test result. These are descriptive changes on the fixed splits, not statistical evidence of superiority. Several content and reconstruction metrics worsened.

## Evidence and experiment

The numeric sources are `reproducibility/raw_logs/srinidhi/vast-part3-tuning-20260929/final-metric-review.json` and the preserved original `task3_gan/srinidhi/outputs/full/evaluation/metrics.json`. The former contains the selection record, both completed training summaries, finalization record, final test metrics, and recorded commands. Metric entries used below have status `computed`; values are rounded for readability.

Both experiments started independently from all four networks in the original 30-epoch checkpoint at update 168,720. They were not trained one after the other on the preceding arm's weights. The original file remained the starting reference and an eligible selection candidate.

| Setting | Frozen continuation recipe |
|---|---|
| Architecture | Two 9-block ResNet generators, two PatchGAN discriminators, 64 base channels |
| Image and batch size | 256 × 256 RGB; batch 1 |
| Training data | 5,624 photos and 240 Monet images; original independent frozen manifests |
| Validation / test | Each split contains 702 photos and 30 Monet images |
| Seed | 2342 in both arms |
| Optimizer | Adam, betas (0.5, 0.999), learning rate 0.00005 |
| Schedule | 10 additional epochs: 5 constant-rate epochs, then 5 with linear decay |
| Updates | 56,240 completed per arm; one epoch is a shuffled pass through the larger domain |
| Smaller domain | Independently sampled with replacement; the domains are unpaired |
| Loss and replay | Least-squares adversarial loss, cycle weight 10, replay pools of 50 images |
| Single changed setting | Absolute identity weight 5.0 versus 2.5 |
| Precision | CUDA BF16 training; recorded evaluation uses the checkpoint directly |
| Restored / reset | Restore all four network weights; reset optimizer, scheduler, replay, RNG, update counter, and best-selection state |
| Validation schedule | Additional epochs 0, 2, 4, 6, 8, and 10; all validation images |

The identity-5 arm first ran a capped 200-update check, then explicitly resumed the same ten-epoch schedule. That check did not replace the schedule with a 200-update decay. The unchanged starting weights were re-evaluated at step zero in both arms, producing identical KID values.

Starting checkpoint SHA-256: `d8b37673ce3cbda3397740863fcde33126eb029a9cc5a7c7cdde9dcf064d9b7e`.

Selected checkpoint SHA-256: `c6775f21d2b6ccc214ae235f366a01f636f0b214e686ca53d2858ecb84d0ec91`.

Frozen protocol SHA-256: `9b9749a20ffcc484428f070867b634524369d2edb3e118d730e696271969da68`.

## Validation selection and the identity comparison

The predeclared score is the equal-weight mean of the two directional validation KID values; lower is better. It is not a mean weighted by the number of generated images. Both arms completed all ten epochs before final selection, and both selected their epoch-eight checkpoint.

| Candidate | Photo → Monet KID | Monet → Photo KID | Mean validation KID |
|---|---:|---:|---:|
| Original, unchanged | 0.0132035064 | 0.0263033448 | 0.0197534256 |
| Identity 5.0, epoch 8 | 0.0121457672 | 0.0202632192 | 0.0162044932 |
| Identity 2.5, epoch 8, selected | 0.0114913839 | 0.0209165701 | 0.0162039770 |

The two arms differ in mean KID by only **0.0000005162**, about **0.0032%** of the control's score. They are effectively tied for interpretation. Identity 2.5 was selected because it has the strictly smaller score under the frozen rule; there was no predefined tolerance treating nearby scores as exact ties. Identity 2.5 scores better for photo-to-Monet, while identity 5 scores better for Monet-to-photo. We cannot conclude that reducing the identity weight caused the overall continuation benefit or is reliably better than retaining weight 5.

The initial discriminator diagnostic showed a sizable real-image train/validation gap for the Monet discriminator: mean raw scores 0.4294 versus −0.0644, with MSE to the real target of 0.3950 versus 1.2028. The photo discriminator's corresponding MSE values were 0.6254 and 0.6665. This was consistent with poor generalization on the small Monet domain, but the scores are not classification probabilities, and a small validation set can also reflect differences in image coverage. The identity ablation does not directly test a discriminator-augmentation remedy. No matched post-tuning discriminator result establishes that this gap was fixed.

The selection record was written at 13:22:55 UTC, before finalization began at 13:23:54 UTC. Its `test_evaluated: false` describes the selection stage. The later `evaluation-final/metrics.json` is the actual test evaluation of that already selected checkpoint; the two records describe successive stages rather than conflicting claims.

## Final test comparison

The original test results had already been viewed before this continuation experiment. The final candidate and checkpoint were selected using validation only; test scores below were calculated afterward for reporting. This test set is therefore not a newly sealed, never-inspected holdout. These results must not be used to choose further hyperparameters, checkpoints, or follow-up tuning runs on the same test set.

The manifest fingerprint matches between original and final evaluation. Photo-to-Monet uses 702 generated images against 30 real Monet references; Monet-to-photo uses 30 generated images against 702 real photo references.

| Direction | Metric | Original | Selected | Observed change |
|---|---|---:|---:|---|
| Photo → Monet | KID ↓ | 0.01111209 | 0.01090441 | 1.87% lower |
| Photo → Monet | FID ↓ | 195.8200 | 195.6824 | 0.07% lower |
| Photo → Monet | Cycle LPIPS ↓ | 0.188557 | 0.183942 | 2.45% lower |
| Photo → Monet | Content cosine ↑ | 0.810303 | 0.805708 | Worse: 0.57% lower |
| Photo → Monet | Cycle L1 ↓ | 0.097505 | 0.098241 | Worse: 0.75% higher |
| Monet → Photo | KID ↓ | 0.01919503 | 0.01712496 | 10.78% lower |
| Monet → Photo | FID ↓ | 188.2377 | 180.9519 | 3.87% lower |
| Monet → Photo | Cycle LPIPS ↓ | 0.236205 | 0.244276 | Worse: 3.42% higher |
| Monet → Photo | Content cosine ↑ | 0.901916 | 0.890011 | Worse: 1.32% lower |
| Monet → Photo | Cycle L1 ↓ | 0.121108 | 0.124390 | Worse: 2.71% higher |

The equal-direction mean test KID changed from **0.01515356** to **0.01401469**, a **7.52% reduction**. The gain is concentrated in Monet-to-photo; photo-to-Monet FID is nearly unchanged. Distribution scores and preservation scores do not show uniform improvement.

| Direction | Distribution metric | Original | Selected |
|---|---|---:|---:|
| Photo → Monet | Precision | 0.601140 | 0.623932 |
| Photo → Monet | Recall | 0.633333 | 0.633333 |
| Photo → Monet | Density | 0.519658 | 0.559259 |
| Photo → Monet | Coverage | 1.000000 | 1.000000 |
| Monet → Photo | Precision | 0.700000 | 0.733333 |
| Monet → Photo | Recall | 0.508547 | 0.521368 |
| Monet → Photo | Density | 0.960000 | 0.900000 |
| Monet → Photo | Coverage | 0.143875 | 0.133903 |

Monet-to-photo density and coverage declined despite improved KID, FID, precision, and recall. PRDC uses k = 5, and its values are sensitive to the small, unequal sample counts. These metrics are not percentages of images proven correct.

Cycle LPIPS compares each input to its cycle reconstruction, not to an arbitrary unpaired target. Content cosine compares frozen ResNet18 features of input and translated image. Neither is a direct human rating of artistic style or photographic realism. FID and KID use frozen Clean-FID Inception features; these evaluators do not generate or touch up the submitted images.

KID used 50 subsets, each limited to 30 images per distribution. The selected test KID subset standard deviations were 0.0055313 for photo-to-Monet and 0.0030957 for Monet-to-photo. These quantify subset variation under this calculation, not independent training-seed uncertainty or a confidence interval for improvement. Only one training seed was used, no significance test was performed, and FID remains biased at small sample sizes. The reported percentage changes should therefore remain descriptive.

## Execution, review, and remaining work

Both summaries identify a single NVIDIA GeForce RTX 5090, PyTorch 2.11.0+cu128, and completed ten-epoch runs with **zero recorded NaN events**. The reported final evaluation metrics are finite and have status `computed`; these checks confirm execution and measurement availability, not visual quality.

| Measurement | Identity 5.0 | Identity 2.5 |
|---|---:|---:|
| Completed updates | 56,240 | 56,240 |
| Cumulative timed update work | 38.22 min | 38.31 min |
| Mean timed update duration | 0.040779 s | 0.040870 s |
| Source images per second during updates | 49.04 | 48.94 |
| Recorded wrapper-session duration | 46.89 min | 48.23 min |
| Peak allocated training GPU memory | 1.466 GiB | 1.466 GiB |
| Recorded NaN events | 0 | 0 |

An update processes two source images, so the image throughput above is twice the paired-update throughput. Timed update work excludes data loading, evaluation, exports, and transfers. The identity-5 wrapper records its resumed session; it excludes the earlier capped check and initial baseline evaluation, although cumulative update time includes the saved updates. Peak allocated training memory is not peak memory across evaluation or CUDA reserved memory. None of these durations is an exact measure of billed instance lifetime.

The finalization record marks final evaluation and export commands completed at 13:25:20 UTC. The exports are neutral generated-image outputs for both directions from the selected checkpoint. The actual class competition submission format has not been verified, and no competition or Canvas submission is claimed.

The fixed validation review used the first six examples in each direction. Its saved assistant review reports retained broad scene composition and subtle differences, with crosshatch-like texture, softened detail, and some horizontal banding still present. It found no obvious new collapse in those twelve pairs, but explicitly did not establish perceptual superiority. This limited assistant review is separate from the required independent two-human audit. The human rating sheets remain blank; no human rating averages or agreement results are available.

The original baseline remains preserved. The complete local backup contains **39,183 files, 6,185,348,875 bytes**, including both trial checkpoints, generated validation images, test evaluation, direct exports, and logs. Its exact inventory and every file's size and SHA-256 matched the frozen remote experiment, with zero differences; verification passed again after promotion to `runs/part3-tuning-vast-backup-20260929/final-full-experiment`. The manifest SHA-256 is `19839c7a8e1f9af7edc5235cf8fc5f3ee415088b064791b7b13f135270e8f3d5`.

Vast instance **53353269** was confirmed stopped at **13:32:43 UTC**: the dashboard showed a Start control and the storage-only rate **$0.023/hour**, replacing the active rate of $0.490/hour. GPU billing ended; its retained disk still incurs storage charges. No instance was destroyed. The monitoring automation was paused after verification.

The separate local publication is `runs/part3-tuned-publication-20260929`. Its notebook, `src/cyclegan.ipynb`, executed all five presentation cells with zero errors, reading saved experiment evidence without retraining. All 3,013 files in its publication manifest were verified. Its selected weights are `checkpoints/selected.pt`; the original member publication was not replaced. Completion receipts are summarized in `reproducibility/raw_logs/srinidhi/vast-part3-tuning-20260929/completion.json`.

The report does not authorize more test-guided tuning or claim the final package is ready for submission while the human audit and submission-format checks remain unfinished.

The research rationale and primary-paper discussion are recorded separately in `report/PART3_FINETUNING_RESEARCH.md`; this report's numerical conclusions come from the experiment artifacts listed above.
