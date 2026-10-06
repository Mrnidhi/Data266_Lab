# Outputs

This directory contains the selected `vast_256_seed3143_v1` run only.

## Training and selection

- `training_history.csv` — 200 epoch-level records.
- `checkpoint_selection.csv` — FID every ten epochs; epoch 190 is best.
- `checkpoint_fid_curve.png` — checkpoint-selection curve.
- `loss_gradient_lr_curves.png` — generator/discriminator losses, gradients,
  cycle/identity loss, and learning-rate schedule.
- `benchmark.json` and `protocol_check.json` — preflight timing and evaluator
  sanity check.
- `selection_manifest.json` — the fixed 300-photo checkpoint subset.
- `selection_preview/` — all 300 deterministic preview predictions used for
  checkpoint comparison.

## Selected epoch 190

- `selected_epoch_190/metrics.json` — consolidated run and selected-checkpoint
  measurements.
- `selected_epoch_190/full_metrics_report.csv` and
  `directional_metrics.csv` — flattened and both-direction reports.
- `selected_epoch_190/submission.csv` — submitted `ID,FID,MiFID` row.
- `kaggle_result.json` — separately records the user-reported leaderboard score.
- `selected_epoch_190/inference_manifest.json` — all 7,038 input/output filename
  pairs and selected-checkpoint identity.
- `selected_epoch_190/translation_cycle_grid.png` — fixed source, translation,
  and cycle-reconstruction examples.
- `selected_epoch_190/human_audit/` — 30 blinded samples and the still-empty
  two-rater rating sheet.

The complete 7,038-image PNG directory and its 827 MB ZIP are intentionally not
duplicated in Git. Their complete filename mapping and count are preserved in
the inference manifest, while the 300 deterministic predictions and audit
samples provide reviewable image evidence. The archive remains in the local
run backup.

`kaggle_metrics_local.json` is a local diagnostic. Its MiFID convention was not
confirmed, so it must not be confused with the submitted row or leaderboard
score.
