# Part 3 — Srinidhi's CycleGAN

Use the [October 4 package](../../reproducibility/packages/part3-20261004/README.md)
for the selected checkpoint, all generated images, executed notebooks, official
CSV and exact training source. The repository's model modules remain readable
entry points; the archive preserves the code version that produced the results.

## Selected result

The unchanged supplied scoring notebook returned **47.56042586442388**:
FID **94.71603584640337**, MiFID **0.4048158824443817**. Lower is better;
the target below 44 was not achieved. See [results](results.md),
[metrics](full_metrics_report.csv) and [failure analysis](failure_analysis.md).

The selected batch-1 checkpoint uses slow EMA at update 221,250. That experiment
completed 370,000 updates; batch-8 and R1 were interrupted by host-memory
exhaustion. Selection considered 192 candidates on class images also present in
training. This is a local class score, not a held-out estimate or Kaggle rank.

Class **A=Monet, B=Photo**. The selected model produced 300 Monet-to-Photo JPEGs
in `pred_A2B` and 7,038 Photo-to-Monet JPEGs in `pred_B2A`. No outputs were edited,
filtered or substituted. The [class protocol](../CLASS_PROTOCOL.md) explains
scoring and the distinction from other GAN metrics.

## Open and reproduce

After `git lfs pull`, extract `reproducibility/packages/part3-20261004/Part3.zip`.
Open its `task3_gan/srinidhi/src/cyclegan.ipynb` with a Python 3.12 environment
and install the package's requirements. It shows recorded results and performs
two CPU translations without training. The unchanged executed scoring notebook
is in the extracted member's `evaluation/` folder.

The verified dataset is reused from `reproducibility/packages/part3-20261002/dataset.zip`.
The package README describes placement and rescoring. For additional metrics in
the main project, install `task3_gan/requirements.txt`.

The selected model has two nine-block ResNet generators and two PatchGAN
discriminators at 256×256. Its batch-1 continuation used learning rate 0.0001,
cycle weight 5, identity weight 0, translation augmentation and EMA. Exact
settings, logs and hashes are inside the package. Member `config.json` describes
the original baseline and smoke profiles; it is not the selected continuation.

`Recovery.zip` preserves the six saved best/last states and the warm-start model.
Read its recovery constraints before resuming: two trials were interrupted,
the original runner has a schedule filename bug, and its older last checkpoint
can contain an outdated best-score record. The working source has tested fixes;
the historical archive was not rewritten. No new GPU run starts automatically.

## Remaining work

The current checkpoint still needs the missing metrics marked in its CSV,
a new blinded human audit and agreement, student review, teammate comparison
and the combined report. Earlier checkpoints' ratings and metrics do not apply.
See the main [submission notes](../../README.md#submission).
