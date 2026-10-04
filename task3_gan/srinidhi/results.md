# CycleGAN results — October 4, 2026

The unchanged supplied scoring notebook reports FID **94.71603584640337**, MiFID
**0.4048158824443817**, and composite **47.56042586442388**. The requested score
below 44 was not reached. This is local notebook evaluation, not a Kaggle rank.

The model has two nine-residual-block, width-64 generators and two PatchGAN
discriminators at 256 pixels. The selected batch-1 continuation used Adam with
LR 0.0001, betas (0.5, 0.999), cycle weight 5, identity weight 0, translation
augmentation for the Monet discriminator, BF16, 1,000 warm-up updates and linear
decay after half the scheduled updates. Generator EMA rates were 0.999 and 0.9999.
The retained checkpoint is the slower EMA at update 221250, not the final update.
Exact recipes and original training source accompany this package.

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
the current human audit, remaining metrics, team report and submissions are pending.
