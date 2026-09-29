# Part 3 fine-tuning: evidence, constraints, and experiment plan

Research checked on 2026-09-29. This is a research and experiment-design note, not a report of improved results. No new model-quality claim follows from the proposed settings. The student must understand, review, and defend the final choices and analysis.

## Recommendation

Keep the independently implemented CycleGAN and compare two controlled continuations from the same saved checkpoint: an unchanged identity-weight control and a lower-identity-weight experiment. First reconstruct a validation baseline and inspect discriminator generalization. Do not introduce several losses, new architectures, and augmentation policies together. The current evidence does not establish that identity regularization is the cause of the remaining painterly Monet-to-photo outputs.

The most credible next improvement, if discriminator overfitting is actually observed, is a separate mild differentiable-augmentation experiment. TTUR is a conditional optimization experiment. CUT, CycleGAN-Turbo, and CycleVAR are useful research comparisons, but they are not drop-in fine-tuning settings for this submission.

## What this project is optimizing

The assignment is an individual implementation and training exercise inside a shared team repository. Its three tasks are character-language-model pretraining, three sentiment classifiers, and unpaired CycleGAN translation. For Part 3, the lab PDF, page 9, requires two generators, two discriminators, adversarial and cycle-consistency losses, evaluation in both directions, a two-rater blinded visual audit, and predictions from the student's own model. It prohibits pretrained/foundation image models from generating or altering submitted images. Page 5 also makes the student's understanding of core decisions and analysis essential. Therefore a visually impressive external image generator is not an acceptable replacement for the trained CycleGAN.

The existing frozen Inception, ResNet, and LPIPS networks only measure saved outputs; they are not part of generation. Keep that distinction explicit. Avoid adding a pretrained perceptual network to the training objective without resolving how the instructor interprets the restriction.

### Verified local baseline

Source: `task3_gan/srinidhi/config.json`, `src/cyclegan.py`, `outputs/full/run_summary.json`, `outputs/full/data_manifest.json`, and `outputs/full/evaluation/metrics.json` under the same member folder.

| Item | Recorded setting or result |
|---|---|
| Training domains | 5,624 photos and 240 Monet images |
| Validation domains | 702 photos and 30 Monet images |
| Test domains | 702 photos and 30 Monet images |
| Data handling | Independent frozen manifests; duplicate-content checks; no paired targets |
| Models | Two 9-block ResNet generators; two PatchGAN discriminators; 64 base channels |
| Images / batch | RGB 256 x 256; batch 1 |
| Training augmentation | Resize to 286 x 286, random 256 x 256 crop, horizontal flip |
| Optimizer | Adam, LR 0.0001, betas (0.5, 0.999) |
| Loss weights | Cycle 10; absolute identity 5 |
| Stabilization | Least-squares adversarial objective; 50-image replay pools |
| Completed training | 30 epochs, 168,720 updates, zero recorded NaNs |
| Schedule | 15 constant-rate epochs, then 15 linearly decaying to zero |
| Selection | Mean validation KID of both directions; best at final epoch, 0.0197534256 |
| Saved starting weights | `task3_gan/srinidhi/checkpoints/best.pt` |
| Recorded starting SHA-256 | `d8b37673ce3cbda3397740863fcde33126eb029a9cc5a7c7cdde9dcf064d9b7e` |

One epoch samples 5,624 examples from each domain: each Monet image is encountered about 23.4 times per epoch on average, with random replacement and augmentation. Thus 30 epochs already imply roughly 703 Monet exposures per image on average. Simply copying another paper's epoch count would conceal a large difference in data exposure. Additional training can overfit the small domain even when no numerical failure occurs.

The published visual-analysis draft reports stronger stylization than reverse photorealism, but the human sheets are still unfilled. That observation is a hypothesis to check on fixed validation panels, not a completed human experiment. Comparing KID 0.01111 in one direction to 0.01920 in the other does not by itself rank their perceptual quality: the target domains, generated counts, and task difficulty differ.

### Correct loss interpretation

Let P be photos, M Monet paintings, G map P to M, and F map M to P. The local implementation computes:

- Adversarial terms: D_M judges G(P); D_P judges F(M).
- Cycle terms: `10 * (L1(F(G(P)), P) + L1(G(F(M)), M))`.
- Identity terms: `5 * (L1(F(P), P) + L1(G(M), M))`.

Identity therefore feeds each generator images already in its **target** domain. It does not directly require a translated Monet image to equal its original painting. Halving this coefficient relaxes a regularizer and may permit larger changes; it does not specifically teach missing photographic texture. The paired cycle paths use both generators, so fine-tune both generators and both discriminators together. Freezing one direction could force the other to accommodate a fixed inverse rather than improve the translation pair.

## Evidence from primary research

The applicability and recommendations below are our project-specific inferences. Paper results on different datasets are not expected gains for this 240-Monet training set.

| Method and primary evidence | Mechanism and relevant evidence | Applicability here | Recommendation and limitation |
|---|---|---|---|
| **Original CycleGAN**: [Zhu et al., ICCV 2017, sections 4, 5.2, 7.1](https://arxiv.org/html/1703.10593v7) | Combines adversarial matching with two-way cycle reconstruction. Uses least-squares GAN loss, a replay buffer, and constant-then-linear LR decay. Identity loss is specifically motivated by preserving color in painting-to-photo translation. | Closest evidence to the model we actually have. The published Monet identity coefficient is 0.5 times the cycle coefficient. Our absolute 5 with cycle 10 matches that relationship. | Keep the existing architecture, replay, and objective. Test identity 2.5 against 5, not against an assumption that identity is harmful. Lower weight risks unnecessary color shifts. Paper epoch counts cannot be transferred without accounting for data size and update count. |
| **TTUR**: [Heusel et al., NeurIPS 2017, sections 2 and 5, appendix A5](https://arxiv.org/pdf/1706.08500) | Gives generator and discriminator separate learning rates. The theory has assumptions; appendix A5 explicitly explains that relative learning dynamics are not determined by LR magnitude alone. | Plausible if our discriminators learn too quickly or too slowly relative to generators. Current code uses one rate for both optimizer groups. | Do not blindly make D twice as fast. First measure real-train, real-validation, and fake discriminator outputs. If a change is justified, vary D's LR alone while holding other settings fixed. TTUR is not a guaranteed convergence fix for this finite run. |
| **DiffAugment**: [Zhao et al., NeurIPS 2020, sections 3.1-3.2 and 4.4-4.5](https://arxiv.org/html/2006.10738v4) | Applies differentiable transforms to real and generated images in the discriminator path, including the generator's adversarial update. Its experiments show why augmenting only real images, or omitting augmentation from G's adversarial path, can fail. | Relevant to the small Monet domain. Existing crop/flip augmentation does not provide the same discriminator regularization. Most published experiments use unconditional or class-conditional GANs, not our exact translation task. | Best conditional follow-up if a train/validation discriminator gap is observed. Start with a modest translation-only policy. Keep cycle/identity comparisons on their original aligned tensors, and keep exported images unaugmented. Strong color transforms or cutout can weaken the style evidence we need D to learn. |
| **ADA**: [Karras et al., NeurIPS 2020, sections 2-3 and 4.3](https://arxiv.org/html/2006.06676v2) | Adjusts augmentation probability using an overfitting heuristic, applying transformations on both D and G adversarial paths. Discusses augmentation leakage and finite-sample limitations. | Addresses a plausible risk here, but its sign-based controller was developed around logistic discriminator logits. Our LSGAN targets are 0 and 1, and our PatchGAN has many correlated spatial scores. | Do not paste the published target 0.6 into this loop. A controller would need calibration for LSGAN and image-level aggregation. Prefer a simpler fixed-policy ablation before adding adaptive state and another selection problem. |
| **CUT / FastCUT**: [Park et al., ECCV 2020, section 3 and section 4.2 ablations](https://arxiv.org/html/2007.15651v3) | Uses multilayer patch contrastive learning to preserve input-output correspondence and enables one-way translation. Identity-style regularization stabilizes some datasets, while removing it helps others. | Offers a different solution to content preservation. Its learned projection heads and contrastive objective are absent from our checkpoint. | Useful future comparison, not the current fine-tune. Replacing the required two-generator cycle model with ordinary CUT would change the assignment architecture. Adding PatchNCE while keeping the cycle model creates a new hybrid requiring a separate design and ablation. |
| **CycleGAN-Turbo, 2024**: [Parmar et al., sections 3.1-3.3, 4.1, 5](https://arxiv.org/html/2403.12036v1) | Adapts pretrained SD-Turbo with LoRA, input conditioning, skip connections, and adversarial/cycle objectives; measures distribution matching and structure preservation separately. The authors also note memory demands of cycle training. | Teaches an evaluation lesson: a model can preserve structure while barely changing style, or match a target distribution while destroying content. | Do not use its generator or pretrained CLIP discriminator for this submission. It relies on foundation-image-model knowledge explicitly excluded from submitted generation. Its one-step inference result is not a training-time estimate for our ResNet CycleGAN. |
| **CycleVAR, ICCV 2025**: [Liu et al., sections 3.2-3.4 and 4.1](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_CycleVAR_Repurposing_Autoregressive_Model_for_Unsupervised_One-Step_Image_Translation_ICCV_2025_paper.pdf) | Adapts pretrained visual autoregressive models, conditions on multiscale source tokens, and relaxes quantization for gradient flow. The experiments use pretrained VAR/Infinity models and CLIP-based discrimination. | A recent alternative demonstrating a different model family, not evidence that its hyperparameters improve our existing networks. | Exclude from the submission recipe for the same pretrained-generation restriction. A newer paper is not automatically a better fit for the rubric, available data, or existing checkpoint. |
| **KID for selection**: [Binkowski et al., ICLR 2018, sections 4-4.1](https://arxiv.org/html/1801.01401v5) | KID estimates squared MMD in Inception features with an unbiased estimator, while FID has finite-sample bias. The paper also explores validation-based LR adaptation. | Supports retaining KID as a selection signal, but 30 real Monet references still produce limited evidence. Unbiased does not mean low variance. | Keep counts, preprocessing, and evaluation randomness identical across candidates. Do not treat repeated subset scores as independent training runs or a generalization confidence interval. |

Read depth: the cited method, implementation, and limitation/ablation sections were consulted, not just titles or abstracts. CycleVAR's full PDF was retrieved directly from CVF and its methods and experiment setup were read. This is a targeted review through relevant 2025 work, not a claim to exhaust all 2026 publications or identify a universal state of the art.

## Diagnostics before a paid training run

1. Verify checkpoint hash, data fingerprints, exact image counts, code revision, software versions, and available disk. Keep the existing final package intact.
2. Run the current checkpoint once on the frozen **validation** split with the current evaluator. Record both directional KID, FID, cycle L1, cycle LPIPS, content cosine, and density/coverage. This checkout retains the summary selection score but not the full historical per-epoch validation metrics, so reconstruct a comparable baseline.
3. Produce a fixed, seeded validation panel set in each direction. Include the complete predetermined set rather than selecting favorable examples. Compare brush texture, color drift, fine detail, scene layout, and artifacts.
4. With networks in evaluation mode and no augmentation, compare D_P and D_M outputs for fixed real training and validation subsets and current generated samples. Aggregate scores per image before summarizing. A large train/validation separation would support overfitting; a low training loss alone would not. These diagnostics must not update weights or consume test examples for selection.
5. Test one full training update, finite gradients in all four networks, checkpoint reload, and a short warmed-up speed measurement on the selected Vast machine. Estimate total time from end-to-end wall time including image loading, and separately measure validation overhead.

## Bounded experiment design

The following values are practical starting hypotheses, not paper-established optima for this dataset.

| Setting | C0: original model | C1: continuation control | C2: identity ablation |
|---|---|---|---|
| Initial weights | Existing best checkpoint | Same four network weights | Same four network weights |
| New training | None | 10 epochs / 56,240 updates | 10 epochs / 56,240 updates |
| Identity coefficient | 5 | 5 | 2.5 |
| Cycle coefficient | 10 | 10 | 10 |
| G and D learning rates | No updates | 0.00005 | 0.00005 |
| Schedule | Original finished schedule | 5 constant epochs, 5 linear decay | Same |
| Adam betas | Existing result | (0.5, 0.999) | Same |
| Architecture and data | Original | Unchanged | Unchanged |
| Precision / batch / replay | Original | BF16 / 1 / 50 images | Same |
| Training seed | 2342 | 2342 | 2342 |
| Validation epochs | 0 | 0, 2, 4, 6, 8, 10 | Same |

Use separate experiment directories and fresh Adam states, LR schedulers, and replay pools. Record parent checkpoint identity and completed parent updates. This is a **warm-start experiment**, because the original scheduler finished at zero and changed recipes must not masquerade as ordinary resume. Restarting optimizer moments is itself part of the recipe; apply it identically to the two arms. Interrupted execution within an arm must resume its own optimizer, replay, RNG, and scheduler state.

Train all four networks. Retain the same seed and sampling procedure across arms to reduce avoidable variation. The two arms together cost 20 additional epoch-equivalents; that does not mean every candidate automatically receives 20 extra epochs.

### Selection and stopping

- Predeclare mean validation KID across both directions as the primary score; C0 remains an eligible winner. Save the best candidate, not just the last update.
- Inspect the directional scores and fixed validation panels so an improvement in one direction does not hide serious degradation in the other. Track content cosine and cycle metrics, but do not optimize them alone: copying the input can score well on preservation while failing translation.
- If the numerical difference is small, assess sensitivity using fixed alternate validation resampling seeds or a paired resampling analysis of saved features. State that 30 Monet references and one training seed limit certainty. The existing KID subset standard deviation is not evidence that a model difference is statistically significant.
- A candidate with lower mean KID but obvious color shifts, content loss, or new artifacts should be reported as a tradeoff, not automatically called better. Genuine human ratings are still required; model-assisted visual inspection is separate.
- Stop immediately on nonfinite loss/gradients. Preserve the last valid checkpoint and record the failure. Complete the fixed trial and stop at its limit; do not extend indefinitely because GAN losses fluctuate.
- If C1 and C2 fail to improve the validation evidence, keep C0. If discriminator diagnostics support overfitting, propose exactly one subsequent DiffAugment trial against the strongest appropriate control. If instability is a balance problem without a generalization gap, investigate separate D/G rates instead. Change one mechanism at a time.

A 10-to-20-epoch extension of a promising arm is a new explicitly documented stage after this comparison, not the default. Since the 10-epoch schedule reaches zero, extension needs its own nonzero schedule rather than blindly resuming a completed scheduler. More epochs are justified only by validation evidence and the remaining cost/time budget.

### Reporting and preservation

After freezing the chosen model, perform the full required test evaluation once for that candidate and retain the original baseline numbers. The existing test results and visual draft have already been viewed, so do not describe them as a never-seen final holdout. Do not use future test scores to steer more trial choices.

Recreate direct inference exports, executed notebook outputs, evidence manifests, and anonymous audit panels for the chosen checkpoint. Preserve raw logs unedited and label each model unambiguously. Complete the two-person visual audit, actual class submission, and teammate comparison separately. Never invent human ratings or claim a local metric is the official Kaggle score.

Back up portable checkpoints and artifacts during training and verify checksums locally before stopping the Vast instance. Record actual billed runtime and rate. The historical 5090 run measured about 1.96 hours of update time for 30 epochs, implying about 39 minutes per 10 epochs at the same update speed; this excludes setup, validation, exports, transfers, and machine-to-machine differences. Use the new benchmark for the spending estimate.

## Questions the student should be able to explain

1. Why are photos and paintings unpaired, and what does cycle consistency add beyond adversarial matching?
2. Why does the photo-to-Monet generator receive real Monet images in its identity term?
3. Why can a lower reconstruction error coexist with weak translation?
4. Why is 5 versus 2.5 a controlled experiment rather than a guaranteed improvement?
5. How do the 240 training paintings and 30 validation paintings constrain generalization and confidence?
6. Why are foundation-model replacements and edited export images outside this lab's submission rules?

Answer these in the student's own words before adopting the final recipe in the report or viva.
