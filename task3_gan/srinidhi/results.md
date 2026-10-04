# CycleGAN results — October 4, 2026

The unchanged supplied scoring notebook reports FID **94.71603584640337**, MiFID
**0.4048158824443817**, and composite **47.56042586442388**. This is local notebook evaluation, not a Kaggle rank.

The model has two nine-residual-block, width-64 generators and two PatchGAN
discriminators at 256 pixels. The selected batch-1 continuation used Adam with
LR 0.0001, betas (0.5, 0.999), cycle weight 5, identity weight 0, translation
augmentation for the Monet discriminator, BF16, 1,000 warm-up updates and linear
decay after half the scheduled updates. Generator EMA rates were 0.999 and 0.9999.
The retained checkpoint is the slower EMA at update 221250, not the final update.
The exact recipe and original training source are in the run manifest linked below.

Batch-1 completed 370,000 updates on an RTX 5090. Batch-8 and batch-8 with R1 were
killed by host memory exhaustion; their recoverable states are at updates 55,000
and 53,000. They were not resumed. Three arms initially shared the GPU, so elapsed
time and utilization do not describe independent single-model benchmarks.

Selection compared 192 candidates using the class scorer on images also used
in training. It is not a held-out comparison and may be optimistically selected.
The full supplied notebook was run afterward on the unchanged exported JPEGs.
Its CSV is authoritative; the companion export scorer differs slightly numerically.
The diagnostic fresh-sample estimate is not an official score or unseen test.

The images are direct model JPEGs in alphabetical source order: A=Monet,
B=Photo; 300 A→B and 7,038 B→A. No images were edited, filtered or substituted.
Read full_metrics_report.csv for metric scope and missing entries. Earlier model
metrics and human ratings cannot be reused for this checkpoint. Student review,
the current human audit, team report and submissions are pending. Some historical
training measurements were not recorded and cannot be recovered through new evaluation.

## Reproduce and trace the result

- Open [src/cyclegan.ipynb](src/cyclegan.ipynb) for the recorded metrics and two fresh CPU translations.
- [src/Part3_Evaluation_Script.executed.ipynb](src/Part3_Evaluation_Script.executed.ipynb) contains the full unchanged class scoring run. [submission.csv](submission.csv) is its output.
- [checkpoints/best.pt](checkpoints/best.pt) is the selected model. Its SHA-256 is `62d80f7ef2752b190fa496b7775b32dd77684bfceeb3b9f4c7356642848abc14`.
- Predictions are in `outputs/pred_A2B/` and `outputs/pred_B2A/`; [training_curves.png](outputs/training_curves.png) plots the unedited training log.
- [Run manifest](../../reproducibility/manifests/srinidhi/part3-push-20261003/manifest.json), [training recipe](../../reproducibility/manifests/srinidhi/part3-push-20261003/configs/b1_c5.json), and [raw logs](../../reproducibility/raw_logs/srinidhi/part3-push-20261003/) connect the model, settings and metrics. `config.json` retains the earlier baseline and smoke profiles, not this continuation recipe.

Training hardware was one NVIDIA GeForce RTX 5090 (32,607 MiB reported), using CUDA 12.8, cuDNN 9.19 and PyTorch 2.11.0+cu128 on Linux. [environment.json](../../reproducibility/manifests/srinidhi/part3-push-20261003/environment.json) records the library versions. The exact host CPU model and per-arm peak memory were not captured in that run manifest; neither is inferred from the local CPU demonstration.

To repeat the supplied class evaluation from the repository root after restoring the shared dataset:

```bash
python scripts/prepare_srinidhi.py --part 3
python task3_gan/srinidhi/evaluate_local.py --output-dir runs/part3-class-evaluation
```

The launcher executes all original notebook cells in their expected folder layout. It writes a new executed notebook and CSV, leaving the saved model, images and recorded results untouched. Model selection used class images seen in training, so this command does not provide a held-out evaluation.

## Image metrics

The fixed post-training audit used all 300 Monet images and 300 photos sampled with seed 2342. Both directions use matching target reference subsets and the unchanged published JPEGs. Cycle L1 and LPIPS compare each input with its cycle reconstruction; PRDC uses the class Inception encoder with k=5. These are diagnostics on training-domain images, not held-out scores.

| Direction | Cycle L1 | LPIPS cycle | Precision | Recall | Density | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Monet → Photo | 0.069930 | 0.238896 | 0.816667 | 0.453333 | 1.140667 | 0.846667 |
| Photo → Monet | 0.078595 | 0.157926 | 0.676667 | 0.713333 | 0.583333 | 0.790000 |

The audit ran on the Apple M1 Max CPU in about 5.3 minutes. [Measured results](outputs/full_metrics_20261004.json) and the [frozen protocol and environment](../../reproducibility/manifests/srinidhi/part3-metrics-20261004/) identify every sampled input, prediction, model and metric weight by hash. The original class score and all published predictions remain unchanged.

To reproduce this audit from the repository root:

```bash
python scripts/prepare_srinidhi.py --part 3
python -m pip install -r task3_gan/requirements.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python task3_gan/srinidhi/src/full_metrics.py evaluate --output runs/part3-full-metrics.json
```

The frozen protocol is already committed; do not rerun the `freeze` command. The evaluation may download standard pretrained metric weights, which are used only to measure results.
