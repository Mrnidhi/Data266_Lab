# Part 3 — Srinidhi's CycleGAN (seed 2342)

The real class data and frozen splits are prepared for Colab training. Full training and final evaluation are not completed yet. No class Kaggle submission is generated or sent automatically.

## Configuration

The full and rehearsal architectures have two 9-block ResNet generators and two 70×70 PatchGAN discriminators, 64 base filters, InstanceNorm, reflection padding, and no dropout. Images are RGB 256×256. Adam uses learning rate 0.0001 and betas (0.5, 0.999); batch size is 1. Full training targets 30 epochs, holding the rate constant for 15 and reducing it linearly for 15. Each epoch is one shuffled pass through the larger training domain; the smaller domain is independently sampled with replacement. Rehearsal stops after 30 updates and does not imply convergence.

The generator objective is the sum of both least-squares adversarial losses, plus `10 × (photo cycle L1 + Monet cycle L1)`, plus `5 × (photo identity L1 + Monet identity L1)`. The identity coefficient here is **absolute 5**; it corresponds to the official reference implementation's relative `lambda_identity=0.5` when cycle weight is 10. Each discriminator minimizes half the sum of its real and replayed-fake least-squares losses. Each domain has a replay pool of 50.

This differs from the teammate's 6-block / 128px model in depth, resolution, learning rate, and schedule. These choices are hypotheses to evaluate, not claims of improved performance.

The Colab A100 configuration uses `precision="bf16"` and `replay_device="device"`. Generator/discriminator forward passes use bfloat16 autocast, while model parameters, Adam state, and loss reductions stay FP32. Native CUDA BF16 support is checked before training. Replay buffers stay on the GPU during training and are serialized as CPU tensors for portable checkpoints. Precision cannot change on resume. CPU smoke tests remain FP32; the CUDA-specific regression test must pass on the allocated GPU before the full run.

## Three run modes

From the project root, with the prepared environment active:

```sh
PYTHONPATH=src python -m lab1.cyclegan train --mode smoke --device cpu --output-dir runs/cyclegan_smoke
PYTHONPATH=src python -m lab1.cyclegan train --mode rehearsal --device cuda --output-dir runs/cyclegan_rehearsal
PYTHONPATH=src python -m lab1.cyclegan train --mode full --device cuda --output-dir runs/cyclegan_full
```

Smoke uses a small 1-block/32px architecture and synthetic noise images; it never downloads data or weights. Rehearsal uses the full architecture. If class data paths are absent, it also uses explicitly labeled synthetic data and reports only pipeline measurements. Full mode refuses to run without actual data. Invalid or partially supplied paths always fail instead of silently switching to synthetic data.

Prepare the actual class data and edit only the chosen mode's `data` object in `task3_gan/srinidhi/config.json`:

```json
{"monet_dir": "/path/to/class/monet", "photo_dir": "/path/to/class/photos", "manifest_dir": "/path/to/manifests", "source_note": "actual verified class dataset/version"}
```

`manifest_dir` must contain `train_monet.txt`, `val_monet.txt`, `test_monet.txt`, `train_photo.txt`, `val_photo.txt`, and `test_photo.txt`. Each line is a relative image path under the appropriate domain directory. Preserve official splits if given; otherwise freeze independent train/validation/test lists before training. Content hashes reject duplicate images within/across splits of a domain. The complete resolved manifest and its fingerprint are saved. The class competition is accessible using the updated Canvas invitation; see [verified class protocol and remaining gaps](../CLASS_PROTOCOL.md). The supplied class archive has been verified and split; the exact scoring script and CSV columns remain unresolved. The public tutorial competition is not assumed equivalent.

Before a long run, measure 100–200 updates on the intended GPU. Estimate compute time as `updates × measured_seconds_per_update`, then allow for data loading, periodic validation, exports, and setup. This is more reliable than predicting time from the GPU name. Check peak memory before changing width; any fallback is a new explicitly recorded experiment.

## Checkpoints and records

Every checkpoint contains all four model states, both optimizer states, both schedules, replay buffers, Python/NumPy/PyTorch RNG states, CUDA/MPS RNG states when available, update index, data fingerprint, and accumulated timing. Resume must preserve the training recipe and manifests. It may increase `max_steps` within the original epoch schedule:

```sh
PYTHONPATH=src python -m lab1.cyclegan train --mode rehearsal --device cuda --output-dir runs/cyclegan_rehearsal --resume runs/cyclegan_rehearsal/last.pt
```

Transfer the complete run, keeping `best.pt` beside `last.pt`, plus the same data and split manifests. Fresh-output resume validates and preserves the previous best checkpoint; missing or inconsistent selected-best artifacts fail explicitly. If a crash leaves logs beyond the last saved update, retain the old run and resume into a fresh output directory. Changing visible CUDA device counts is supported, but cross-device continuation is not guaranteed to be bitwise identical. See [portable run instructions](../../COMPUTE_PLAN.md).

`training_log.jsonl` records raw directional cycle/identity losses, adversarial losses, discriminator losses, gradient norms for each network, NaN events, learning rate, and update timing. `run_summary.json` records counts, parameters, memory, runtime, and provenance. Throughput counts two source images per update; it excludes loading/evaluation and is not the number of generator forward passes. CUDA peak allocated training memory is measured; CPU/MPS memory is explicitly unavailable. Fixed input → translation → reconstruction grids are saved. Non-finite losses/gradients stop training before applying the update.

`best.pt` is selected only from genuine **validation** KID averaged across both directions, every five epochs. If required metric dependencies or weights are unavailable, no best model is invented. `last.pt` remains a resumable checkpoint, not a claim of best quality. Always inspect fixed grids and content metrics alongside KID.

## Held-out evaluation

The PDF's member-level entry point is available. From the repository root:

```sh
.venv/bin/python task3_gan/srinidhi/evaluate_local.py --mode full --device cuda --checkpoint task3_gan/srinidhi/checkpoints/best.pt --split test --output-dir task3_gan/srinidhi/outputs/evaluation-final --report task3_gan/srinidhi/full_metrics_report.csv
```

This exports the full directional metrics CSV while preserving unavailable values and reasons. It refuses to overwrite a populated report. The checked-in header-only CSV is a pending-results placeholder. The checkpoint and actual data must exist before this command runs.

```sh
PYTHONPATH=src python -m lab1.cyclegan evaluate --mode full --device cuda --checkpoint runs/cyclegan_full/best.pt --split test --output-dir runs/cyclegan_test
```

Use a fresh evaluation directory. Evaluation reads only its explicitly named validation or test manifests and processes both directions. The optional dependencies `clean-fid`, `lpips`, `prdc`, and `torchvision` are lazy-loaded; loading pretrained metric weights requires `evaluation.allow_metric_downloads=true`. Full mode enables this. They are frozen evaluators, not pretrained generators. The interpretation that frozen evaluation networks are permitted should be checked against any broader instructor prohibition.

- FID and KID compare generated target-domain images with real held-out target-domain images using the same Clean-FID Inception features/preprocessing. Record real/fake counts, KID subsets and subset size. Small Monet sets make distributional estimates uncertain; subset SD is not training-seed uncertainty.
- Generative precision, recall, density, and coverage use those same features and the configured neighborhood size. They are distinct from sentiment-classifier precision/recall.
- Cycle L1 and AlexNet LPIPS compare each source with its **own cycle reconstruction**. An unrelated real Monet image is not a paired ground truth.
- Content cosine compares a source and its direct translation using frozen torchvision ResNet18 ImageNet features. This is explicitly ResNet18 pooled-feature cosine, not DINO structural distance.
- Missing/failed optional metrics have `status: unavailable`, `value: null`, and a reason. Synthetic data cannot produce purported class-quality FID/KID scores. Inspect every unavailable entry before preparing the final report.

Metric weights can be prepared separately before renting a GPU:

```sh
python scripts/prefetch_metric_weights.py
python scripts/prefetch_metric_weights.py --offline
```

The script verifies real metric APIs on synthetic fixtures and writes `verification/metric_dependencies.json`; these checks are not image-quality results. It caches Clean-FID Inception, LPIPS's AlexNet backbone, and ResNet18 weights under `.cache/metric_weights/` (about 369 MiB). LPIPS's calibration weights come with its installed package. Copy that cache to the same project-relative directory on the GPU machine, or set `LAB1_METRIC_CACHE` to its new location, then run the offline check. The optional cache is separate from the small source-code handoff. SciPy is pinned to **1.16.3** because Clean-FID 0.1.35 uses the `sqrtm(..., disp=False)` API removed in newer SciPy.

## Human audit and direct export

Evaluation creates two **blank** rater CSVs and anonymous original/output panels. The lab asks for 30 fixed samples and evaluation in both directions; this pipeline conservatively prepares up to 30 per direction, reporting actual counts if fewer exist. Two humans independently rate style, content, and artifacts from 1 to 5. For artifacts, 5 means no visible artifacts. Hide the private mapping file and student/model labels. Use the same input subset across teammates. Blank ratings are not accepted as results.

```sh
PYTHONPATH=src python -m lab1.cyclegan audit-score --rater-one runs/cyclegan_test/human_audit/rater_1.csv --rater-two runs/cyclegan_test/human_audit/rater_2.csv --output-dir runs/cyclegan_audit_scored
```

Agreement includes percent exact agreement and quadratic-weighted Cohen's kappa by criterion and direction. Undefined kappa is reported as null, not zero.

```sh
PYTHONPATH=src python -m lab1.cyclegan export --checkpoint runs/cyclegan_full/best.pt --device cuda --direction photo_to_monet --input-dir /path/to/verified/class/inference/photos --manifest /path/to/inference_photo_manifest.txt --output-dir runs/cyclegan_direct_export
```

Export writes direct-inference RGB PNG images and a checkpoint-hashed manifest into a neutral folder. Synthetic checkpoints are refused. This is **not** a Kaggle-ready package: verify the actual class requirements before choosing filenames, image count/resolution, JPEG/PNG conversion, and ZIP/CSV packaging. No account operation or submission occurs.

The required prediction directories are `outputs/pred_A2B/` and `outputs/pred_B2A/`. For this member, A means Photo and B means Monet. Use a fresh run subdirectory under the appropriate direction (for example `outputs/pred_A2B/run-001`) because export rejects a nonempty destination. `submission.csv` is intentionally pending until the real competition schema and inference outputs are available.

## References

- [Original CycleGAN paper](https://arxiv.org/abs/1703.10593) and [official loss semantics](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py).
- [Clean-FID](https://github.com/GaParmar/clean-fid), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [PRDC](https://github.com/clovaai/generative-evaluation-prdc).
- [The FID Lottery, June 2026](https://arxiv.org/abs/2606.20536): avoid treating single-seed small score gaps as architectural proof. Its diffusion-specific numeric thresholds do not transfer to this lab.
- [Rethinking FID through reference geometry, May 2026 workshop](https://arxiv.org/abs/2605.29335): interpret distribution metrics with coverage, content, and human inspection.
