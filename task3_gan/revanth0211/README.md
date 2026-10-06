# Part 3: CycleGAN — Revanth

This folder contains my independent Monet ↔ photo CycleGAN run. I trained two
ResNet generators and two PatchGAN discriminators from random initialization on
an NVIDIA GeForce RTX 4090. The current run uses 256 × 256 crops, seed 3143,
200 epochs, 512 update pairs per epoch, and all 7,038 photos as the sampling
pool. The fixed-subset checkpoint evaluation selected epoch 190.

Only the notebook for this selected run is published here:
[src/Part3.ipynb](src/Part3.ipynb). Older exploratory notebooks and older
128-pixel runs are intentionally excluded.

## Current result

| Measurement | Value |
|---|---:|
| Selected epoch | **190** |
| Fixed-subset selection FID | **103.877408** |
| Submission CSV FID | **107.211839** |
| Submission CSV MiFID | **0.418532** |
| Submission composite, `(FID + MiFID) / 2` | **53.815186** |
| Latest leaderboard score reported after upload | **-53.8151** |
| Full-photo local diagnostic FID | 88.253025 |
| Full-photo local diagnostic MiFID | 88.253025 |

The local full-photo MiFID uses the notebook's provisional rule and is not
presented as the Kaggle value. The accepted submission row and the reported
leaderboard score are recorded separately in
[outputs/selected_epoch_190/submission.csv](outputs/selected_epoch_190/submission.csv)
and [outputs/kaggle_result.json](outputs/kaggle_result.json).

## Model and training design

Each generator uses two downsampling stages, nine residual blocks at base width
64, and two upsampling stages. Each discriminator is a PatchGAN. Training uses
least-squares adversarial loss, cycle-consistency weight 10, identity weight 5,
Adam with learning rate 0.0002 and betas (0.5, 0.999), batch size 1, a replay
pool of 50 images, and AMP. The learning rate stays constant through epoch 100
and then decays linearly through epoch 200.

Checkpoints were evaluated every ten epochs on the same 300-photo subset. FID
fell from 143.945067 at epoch 10 to 103.877408 at epoch 190, then rose slightly
to 104.728361 at epoch 200. The selection rule therefore retained epoch 190
instead of simply using the last epoch.

## Folder map

- `src/Part3.ipynb` — the only published Part 3 notebook and the source for the
  selected run.
- `outputs/` — raw metrics, training history, checkpoint-selection evidence,
  plots, 300 deterministic preview predictions, the submission CSV, and the
  30-sample blinded human-audit packet.
- `checkpoints/` — selected-checkpoint status and manifest.
- `data_processed/` — dataset-count and preprocessing documentation; raw images
  are not duplicated in this member folder.
- `metrics_report.csv` — compact required metrics in one file.
- `full_metrics_report.csv` — full flattened configuration and run record.
- `failure_analysis.md` — concrete limitations observed in saved outputs.
- `results.md` — architecture, results, runtime, and interpretation.

## Reproduce

Place `monet_jpg/`, `photo_jpg/`, and `real_stats.npz` under a sibling `data/`
folder, open the notebook from its own working directory, review the preflight
and benchmark, then explicitly set `RUN_TRAINING=True` before starting the long
run. The notebook was designed for a persistent terminal session:

```bash
cd task3_gan/revanth0211/src
tmux new -s part3
jupyter nbconvert --to notebook --execute Part3.ipynb \
  --output Part3_executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

The published notebook source does not fabricate cell outputs. The downloaded
run evidence is preserved under `outputs/`. Training again will create a new
run folder and may not reproduce the exact floating-point result on different
software or hardware.

## Important remaining item

The selected epoch-190 file is named `best_fid.pt` in the inference manifest,
but that checkpoint was not included in the downloaded output archives and is
not present on this computer. An older checkpoint is not substituted because it
would not match this result. Download `best_fid.pt` from the completed Vast run,
then add it through Git LFS and update `checkpoints/manifest.json` with its size
and SHA-256 before calling this folder fully archival.

The two-rater visual audit is also still blank. The 30 blinded samples and
rating sheet are included, but no human score is invented.
