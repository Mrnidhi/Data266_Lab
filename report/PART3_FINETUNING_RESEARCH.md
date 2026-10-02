# Part 3 fine-tuning: evidence, constraints, and experiment plan

Research checked on 2026-09-29. This is a research and experiment-design note, not a report of improved results. No new model-quality claim follows from the proposed settings. The student must understand, review, and defend the final choices and analysis.

## Follow-up research checked October 2, 2026

**Prospective access-window amendment, 07:52 Pacific:** the user subsequently set a hard 11 AM lab-access deadline. The five-candidate policy in `runs/part3-ema-study-20261002/deadline_amendment.json` was frozen at update 8,417, before any nonzero validation output existed. Eligible endpoints are now the unchanged source plus raw/EMA at epoch four in each learning-rate arm. Both endpoints use four constant-rate epochs of the unchanged original eight-epoch schedule. The already-running first arm finishes its original invocation, but all epoch-eight scores are excluded unconditionally; the second stops at step 22,496. This completes the amended matched-endpoint comparison only, not the original eight-epoch study. The unchanged directional guardrail and source tie preference still apply. Training stops by 10:15 (10:17 hard guard for metric overrun), class scoring by 10:35, and finalization by 10:50. No incomplete candidate subset is selected. The six-hour proposal below is retained as the superseded design, not the current allowance.

The sections after this update preserve the September 29 plan as historical context. The user's subsequent October 2 request to rework the model for a top-five target is being acted on immediately; it supersedes the earlier ambiguous “at 12” scheduling discussion. A six-hour cap is the assistant's chosen limit for this first new round, not a user-confirmed budget or permission for indefinite computation. Implementation and verification started around 07:10 Pacific. No new recurring automation is being created. Top five remains an objective, not a predicted or guaranteed result.

### What the completed study actually supports

The verified incumbent is control epoch 8 from `runs/part3-translation-study-20261001`, SHA-256 `5261abcdfe3bdbe67c7c98551bd4134a09947136b4e344d27ecb9e81027c9c3f`. Its mean validation KID is 0.015693356920965018, versus 0.01620397192891687 for the earlier source. The equal-direction mean improved 3.15%, but Photo-to-Monet KID worsened slightly. The mild Monet translation arm never beat its initial source. Its failure is evidence against repeating that exact setting, not against every form of regularization.

Both trajectories initially worsened after restarting Adam at LR 0.00005. Control's best validation point occurred after decay to about 0.00002; its last checkpoint was worse despite lower reconstruction losses. A gentler restart is therefore a practical hypothesis. These observations do not isolate learning rate causally, because training duration and state also changed. A matched comparison is needed.

The current implementation has the expected detached replay, discriminator freezing for generator gradients, two generators/discriminators, and resize/crop/flip data augmentation. No confirmed training bug was found. The final class scores, measured after selection, are recorded in `task3_gan/CLASS_PROTOCOL.md`; they must not become feedback for selecting the next hyperparameters. Keep the complete `runs/part3-final-package-20261001` package and ZIP immutable.

### Research priorities and limits

| Priority | Evidence and mechanism | Proposed use in this project |
| --- | --- | --- |
| Gentler continuation | Local validation worsened early after the LR restart. The [authors' CycleGAN guidance](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md) also cautions that loss curves alone do not establish image quality. | Compare LR 0.00005 and 0.00002 from the same incumbent, holding all other training choices fixed. These numerical choices are project hypotheses, not published optimal Monet settings. |
| Generator exponential moving average (EMA) | [Yazici et al., ICLR 2019](https://arxiv.org/html/1806.04498v2) study temporal parameter averaging in GANs. Their EMA improves several GAN experiments and operates outside the training optimization. Their theory does not guarantee convergence or improvement for this CycleGAN. | Maintain a separate FP32 average of each generator along each new trajectory; compare raw and EMA candidates at fixed validation points. Initialize from the same source, update after optimizer steps, and preserve EMA state on resume. Do not approximate temporal EMA by averaging unrelated existing best/last endpoints. |
| Generator/discriminator learning-rate balance | [Heusel et al., NeurIPS 2017](https://arxiv.org/abs/1706.08500) motivate separate generator/discriminator rates under stated assumptions. The result does not prescribe a universally correct rate ratio. | First repeat the same train/validation discriminator diagnostic on the current incumbent. The available discriminator-gap receipt describes the older source. A slower Monet discriminator is a later isolated hypothesis, not an automatic fix or a change to combine into the LR comparison. |
| Identity loss | [Zhu et al., ICCV 2017](https://arxiv.org/html/1703.10593v7) motivate identity loss for preserving color in painting-to-photo translation. The local coefficient is absolute, unlike the authors' relative flag. | Keep absolute identity 2.5 and cycle 10 for the next LR test. A later 2.5-versus-1.25 ablation could test stronger style change, but risks color/content drift and is not justified solely by low class scores. |
| Upsampling artifacts | [Odena, Dumoulin and Olah, 2016](https://distill.pub/2016/deconv-checkerboard/) explain artifact-prone transposed convolution and demonstrate resize-then-convolve alternatives. Our generator uses kernel 3, stride 2 transposed convolutions. | A plausible architectural contributor, not a diagnosed cause of all painterly texture. First inspect fixed validation examples and, if needed, prespecified spatial-frequency diagnostics. Changing the decoder requires new weights/adaptation and a separate architecture experiment; do not silently reinterpret existing kernels or smooth exported images. |
| Additional discriminator regularization | [Zhao et al., AAAI 2021](https://arxiv.org/abs/2002.04724) show both benefits and possible artifacts from consistency regularization in other GAN settings. | Defer until simpler continuation/EMA evidence is available. Adding several new penalties at once would obscure the cause of any change. Ordinary CUT or pretrained diffusion replacements also do not meet the assignment's stated two-generator cycle architecture/generation requirements. |

### Bounded experiment being implemented

Use a new protocol, runner and output directory rather than editing the completed experiment. Preserve the current source as an eligible candidate. Proposed design for a six-hour round:

- Two independently warm-started arms from the same incumbent: LR 0.00005 versus 0.00002 for both G and D. Eight epochs per arm, four constant-rate followed by four linear-decay epochs, with fresh matched Adam/replay/RNG states. Seed 2342, BF16, batch 1, original manifests, cycle 10, identity 2.5 and no added translation augmentation remain fixed.
- Maintain both ordinary generators and EMA generators with one declared decay, initially proposed beta 0.999. This beta has an effective recent-update scale of roughly 1,000 steps; it is a practical chosen setting, not a paper-established optimum for batch-one CycleGAN. Do not search many decays after observing results.
- Validate the source once, then raw and EMA candidates at epochs 4 and 8 in each arm. This bounds selection to eight new candidates plus the unchanged source. EMA evaluation must substitute both generators consistently while leaving training state untouched. Save raw optimizer/replay/RNG/EMA state for exact interruption recovery and export explicitly identified selected generator weights.
- Primary selection stays equal-direction mean validation KID on the existing frozen validation identities, with the unchanged source winning exact ties. Before selecting by the mean, require neither directional KID to exceed the incumbent's rechecked value by more than 1e-8. This predeclared numerical non-regression screen is a conservative engineering choice, not a confidence interval or proof of perceptual quality. Report both directional values, content cosine, cycle LPIPS, coverage and the same first-six visual panels. Obvious collapse or nonfinite values fail the run. Small numerical gains should be described as uncertain rather than as overall perceptual superiority.
- Existing measured ten-epoch arm times are about 1.80 and 1.86 hours. Two eight-epoch arms suggest roughly three hours of training; allow additional time for EMA validation, implementation checks, scoring and packaging. This is an estimate, not a deadline guarantee. No automatic third arm or unlimited retries are included. Stop at the agreed time limit and retain the incumbent if no acceptable improvement is found.

This experiment needs implementation before it can run: the old `scripts/tune_cyclegan.py` hardcodes LR 0.00005 and a ten-epoch schedule, and its finalizer hardcodes the completed translation study. Editing an input config alone would not perform this proposal. New isolated trainer/runner files are being added without changing the old implementation or packaged evidence. Required checks include unchanged legacy behavior, matching raw/EMA initialization, correct EMA update arithmetic, uninterrupted versus resumed equivalence including EMA state, and scoring/export of exactly the selected weights.

The current incumbent's new diagnostic at `runs/part3-ema-readiness-20261002/incumbent_discriminator_diagnostics.json` reports Monet real-target MSE 0.4085 on 128 training images versus 1.7595 on 30 validation images; Photo MSE is 0.7919 versus 0.7970 on 128 images each. This continues the descriptive Monet train/validation gap and does not establish its cause. No extra arm was added in response. `input_verification.json` in that directory verifies all 7,338 original image hashes, six frozen split hashes, supplied notebook and incumbent checkpoint before the new study.

### Evaluation integrity for further work

Thirty Monet validation images give limited information. [Binkowski et al., ICLR 2018](https://arxiv.org/html/1801.01401v5) explain FID bias and the unbiased KID estimator; unbiasedness does not remove variance or the optimism from choosing among many candidates. Current KID subset standard deviations are not training-seed confidence intervals. The smaller 30-image side is reused across subsets, so increasing subset count does not supply new Monet evidence.

The [Clean-FID study](https://arxiv.org/abs/2104.11222) shows that resizing and compression can change metric values. Therefore keep the instructor's metric definitions, alphabetical order and fixed JPEG quality-95/subsampling-0 export unchanged. If an instructor-style feature diagnostic is added, predeclare it on validation-only images, label its smaller sample count and never call it the official class score. It must not replace the primary metric retroactively. No JPEG-quality search, reordered outputs, cherry-picked images or test/leaderboard-driven selection is part of this plan.

After selecting and freezing a candidate, evaluate the supplied class scorer once for reporting and regenerate a separate verified package only if needed. Existing test outputs have already been inspected; future test reporting is not a newly sealed holdout. Two independent human ratings, teammate comparison, student understanding and official submission remain separate requirements. No finite search can establish a globally best model or guarantee top three/10 bonus points.

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
