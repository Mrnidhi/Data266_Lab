# Continue DATA266 Lab 1 — Part 3 on my RTX 5090

I want you to continue improving my own CycleGAN for DATA266 Lab 1, Pair 49, using the RTX 5090 available to me for approximately six hours. Aim for the strongest defensible class result; do not promise first place. This is a continuation of completed experiments, not a request to restart the entire lab. Work in the current VS Code repository, `Mrnidhi/Data266_Lab`.

## Start by inspecting the current machine

1. Read `VSCODE_HANDOFF.txt`, `PART3_NEXT_GPU_PLAN.txt`, `task3_gan/CLASS_PROTOCOL.md`, `report/PART3_FINETUNING_RESULTS.md`, and `report/PART3_FINETUNING_RESEARCH.md`. Follow any applicable repository instructions. Check the Git branch and existing changes; preserve unrelated work and my teammate's files.
2. Verify the actual GPU, CUDA/PyTorch compatibility, free VRAM, other GPU workloads, disk space, and remaining access time. Do not assume this computer has the source Mac's files, paths, credentials, or chat history. If the GPU is on this workstation, use it directly; SSH is only needed for a remote machine.
3. Locate the selected checkpoint, frozen data manifests, raw images, and `real_stats.npz`. Git does **not** include model weights, datasets, `runs/`, or my edited human-review CSV. Ask for genuinely missing files or connection details after checking what is available. Do not request passwords/private keys or copy credentials into the repository.
4. Tell me briefly what is present, what is missing, and the measured plan. Continue independent preparation while resolving missing inputs. Do not launch duplicate jobs or silently install over a working CUDA environment.

## Completed work and selected parent

- Parts 1 and 2 have already finished. Do not retrain or modify them for this task.
- Part 3 completed 30 original epochs, followed by two independent ten-epoch continuations comparing identity weights 5 and 2.5. The existing model has two nine-block ResNet generators, two PatchGAN discriminators, base channels 64, 256×256 images, and batch size 1.
- The selected checkpoint is `runs/part3-tuned-publication-20260929/checkpoints/selected.pt`.
- Its SHA-256 must be `c6775f21d2b6ccc214ae235f366a01f636f0b214e686ca53d2858ecb84d0ec91`.
- Identity 2.5 at continuation epoch 8/update 44,992 was selected using mean directional validation KID. Its score was **0.0162039770**. Identity 5 scored **0.0162044932**: these are effectively tied, so reducing identity weight is not a proven cause of improvement.
- Selected local held-out results were Photo→Monet FID **195.6824**, KID **0.01090441**, and Monet→Photo FID **180.9519**, KID **0.01712496**. These are **not official Kaggle scores**. Several content/reconstruction metrics worsened.
- Frozen splits are Photo 5,624/702/702 and Monet 240/30/30, train/validation/test, seed 2342. Preserve those manifests and the original checkpoint. The existing test results have already been inspected; do not use them to choose new settings or checkpoints.
- Intended data locations are `task3_gan/data/monet_jpg`, `task3_gan/data/photo_jpg`, and ignored processed manifests under `task3_gan/srinidhi/data_processed/part3_manifests`. Tracked membership/hash snapshots are in `reproducibility/manifests/srinidhi/part-c-source-snapshot/`. Rebase machine-specific file paths without changing membership or resplitting. The supplied archive, `data-266-fall-2026-gan-image-style-transfer.zip`, has SHA-256 `5060b62d4a6a0194de10b849ea43b649f308ae516f0bf659b62ee404ca153e1d`. Do not silently substitute the original baseline if the selected checkpoint is missing.
- `src/lab1/__init__.py` intentionally extends the import path to member source folders. The implementation is `task3_gan/srinidhi/src/cyclegan.py`; `scripts/tune_cyclegan.py` is the continuation runner. Do not treat the absence of `src/lab1/cyclegan.py` as a missing implementation.

## Resolve the class evaluation before claiming a score

Competition: https://www.kaggle.com/competitions/data-266-fall-2026-gan-image-style-transfer

On October 1, this account showed no submissions. The leaderboard leader was Team 41 at **−41.4884**. These are dated observations; recheck if current standings matter. The Overview describes an average of FID and its class-specific MiFID proxy, with lower values better, but displayed leaderboard values are negative. The exact transformation, evaluated directions, feature preprocessing, and CSV schema remain unverified. Do not invent them or substitute local Clean-FID scores.

The official evaluator and example `submission.csv` were not found in the repo, supplied ZIP, or Kaggle Code tab. Canvas required sign-in. Locate the instructor-provided script/template through available course materials; ask me for it if unavailable. `real_stats.npz` contains `mu_real`, `sigma_real`, and 300 real feature vectors, but their shapes do not establish the full scoring procedure. If the script remains unavailable, continue useful internal validation work and clearly state that the official score is still blocked.

The instructor clarified the export protocol here:
https://www.kaggle.com/competitions/data-266-fall-2026-gan-image-style-transfer/discussion/744502

- Class **A = Monet, B = Photo**. Translate all 300 Monet images, including training images, into class `pred_A2B`. This class evaluation is separate from our held-out study.
- Existing local exports use the **opposite A/B convention**. For new class folders, use **`G_monet_to_photo` → `pred_A2B`** and **`G_photo_to_monet` → `pred_B2A`**. Verify the required Photo input list with the scorer; do not assume the deduplicated local list is the class list.
- Generate fresh 256×256 RGB JPG outputs directly from the selected networks. Preserve historical PNG exports and their documented directions; do not rename checkpoint keys or edit generated images.
- Use our own trained CycleGAN lineage. Do not substitute pretrained/foundation image generators, copied images, handpicked outputs, or manual touch-ups. Pretrained evaluation encoders serve only the documented evaluation role.

## Proposed next experiment — implement and test before full training

First reproduce the selected checkpoint's validation score and recheck unaugmented discriminator diagnostics. Earlier Monet-discriminator real-target MSE was approximately **0.395 on training images versus 1.203 on validation images**, suggesting a generalization gap. It has not been established that the selected continuation fixed it.

Prepare a matched **unchanged continuation versus translation-only DiffAugment** comparison. Augmentation is proposed, not already implemented or proven better. Review the primary paper and existing research notes:
https://arxiv.org/abs/2006.10738

The existing `scripts/tune_cyclegan.py` hard-codes the old `identity5`/`identity2p5` arms and seed 2342. Create or extend a separately versioned protocol/runner for this new experiment, with tests. Do not invoke the old comparison and label it a DiffAugment or second-seed run. Preserve reproducibility of the completed September 29 protocol.

- Start both arms independently from all four networks in the same selected parent. Reset optimizer, schedule, replay pools and continuation RNG equally; retain the parent as an eligible winner.
- Hold other settings fixed: identity 2.5, cycle 10, replay 50, Adam betas (0.5, 0.999), learning rate 0.00005, batch 1, BF16 after compatibility verification.
- Proposed budget per arm: **56,240 updates**, first 28,120 at constant LR, then linear decay. Validate at step zero and every 11,248 updates. A short rehearsal must retain the full intended schedule rather than decay the LR over just its few steps.
- Treatment: mild differentiable translation on the Monet discriminator's real/fake inputs, including the generator's adversarial gradient path. A maximum ±16-pixel shift is an initial hypothesis, not a proven optimum. Leave cycle/identity comparisons unchanged and store replay images unaugmented. Give augmentation its own checkpointed RNG to preserve matched data/replay streams.
- Test gradient flow, consistent real/fake treatment, unchanged control behavior, and checkpoint/resume behavior. Then run a separate approximately 200-update GPU rehearsal; check finite losses, speed, utilization and memory before full runs.
- Freeze a validation-only selection rule before launching. If the class scorer is verified, align the validation calculation where possible while retaining the independent split. Otherwise use the existing mean directional validation KID and label it internal evaluation. Record both directions and content-preservation metrics; do not repeatedly tune against the leaderboard or existing test results.
- If the first comparison improves validation without clear deterioration in fixed image panels, and time permits, repeat **both arms** with a second continuation seed. This checks continuation randomness, not independent original training seeds. Keep the existing model if new runs do not improve it.
- Do not combine multiple architecture/loss changes in this session. Resize-convolution is a possible later controlled experiment for grid artifacts, not a confirmed remedy or a drop-in checkpoint replacement.

## Time, progress and completion

Our earlier RTX 5090 trials took about **38 minutes of timed update work** per 56,240 updates and **47–48 minutes per recorded wrapper session**. Those measurements exclude some setup/finalization and do not predict this machine exactly. Benchmark the actual recipe and revise the time budget.

Reserve roughly 30–45 minutes for readiness/baseline, 2–2.5 hours for the first comparison, up to 1–1.5 hours for a justified repeat, and at least the final hour for evaluation, exports and verified backups. Reduce the training scope if the measured speed or remaining access window requires it. Never sacrifice a verified final checkpoint for one more unfinished experiment.

Keep logs visible and give me the exact command to follow progress. Save portable checkpoints with the required optimizer/scheduler/RNG/replay state and record actual completed updates. Use separate dated run directories and preserve baseline artifacts. Select using validation, inspect fixed outputs, then report final metrics, direct exports, provenance and any unresolved issues honestly.

I completed one personal human review of the September 29 model with AI assistance for writing. Preserve `My_review.csv` and its provenance. Those ratings do not apply to a new model; do not invent another human review or agreement score. The rubric's second independent review remains outstanding.

Use my available GPU. Do not rent another GPU, buy credits, restart old cloud instances, or destroy stored instances. At the end, stop this task's training processes after verifying backups; do not shut down an entire shared lab workstation. Prepare any Kaggle submission as a reviewable artifact and ask before the actual upload. Do not claim training, scoring, backup, submission, or shutdown is complete without checking the corresponding evidence.

Begin with the local readiness inspection now, then proceed with the authorized implementation, checks and GPU work as prerequisites become available. Ask only for information or actions you genuinely cannot obtain or complete yourself.
