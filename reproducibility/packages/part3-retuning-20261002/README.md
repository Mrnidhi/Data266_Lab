# Part 3 retraining checkpoint backup

This archive preserves all eight checkpoints from the October 2 LR/EMA study, plus its original warm-start source, the original recipes, frozen splits, training logs, validation receipts, selection/amendment records, and matching source code. It does not start training or change the selected result.

## Which checkpoint to use

- For a NEW fine-tuning experiment, the selected model is `runs/part3-ema-study-20261002/arms/lr5e5/candidates/step_0022496/ema.pt`, SHA-256 `0c2eab37c42062bbd7b3b2c5c5c097adc949535ff77a9f805ad89842f6f8073b`. It has both selected generators and the corresponding discriminators. Warm-starting retains these weights but starts fresh optimizer, schedule, replay, random and averaging state. This is not an exact continuation of optimizer state.
- `runs/part3-ema-study-20261002/arms/lr5e5/last.pt` preserves complete training state at 44,992 updates / eight epochs. This is the last online state, not the selected epoch-four EMA model. Its original eight-epoch schedule is finished; the unchanged trainer refuses further steps beyond that schedule.
- `runs/part3-ema-study-20261002/arms/lr2e5/last.pt` preserves complete state at 22,496 updates / four epochs. Continuing within the original eight-epoch recipe would require raising/removing the recorded step cap and satisfying the trainer's original integrity checks.
- Other raw/EMA candidate checkpoints are preserved for recovery. Epoch-eight results were excluded from the completed selection and this backup does not change that decision.

Both `last.pt` files contain raw network weights, Adam optimizers, learning-rate schedulers, replay buffers, RNG state, paired generator EMA and progress. Full resume also needs the matching saved run directory, recipe, data manifest, complete training log, source checkpoint and completed validation receipts. Their absolute original paths are intentionally preserved. Restoring on another machine needs careful path handling; this archive is not a promise of one-command portable or bitwise identical resume. Do not silently rewrite old evidence or bypass hash checks. The old deadline coordinator/budget are historical and should not be restarted as a new experiment.

## Restore the data and use the files

Retrieve Git LFS files with `git lfs install` and `git lfs pull`, then extract `Part3_Retuning_Backup.zip` into a fresh directory. It contains one `Part3_Retuning/` root with project-relative paths. Keep the immutable backup intact and work in a separate copy.

The original dataset and completed results are already backed up separately:
- https://github.com/Mrnidhi/Data266_Lab/tree/main/reproducibility/packages/part3-20261002

Restore the verified dataset and the six frozen split lists before any future training. Keep the original architecture compatible with the checkpoint. For a new tuning round, declare its validation selection and time budget before results, use a fresh output directory, and run final class scoring only after selection. No training is launched by this backup.

## Scope

All nine checkpoint files and every preserved study text/JSON/JSONL receipt are hash-listed in `MANIFEST.json`. Raw dataset images and PNG preview/intermediate images are omitted: the dataset is in the separate verified backup and previews can be regenerated from the included checkpoint files. Completed submission images and the executed results notebook remain in the separate final Part 3 package. Temporary lock files and machine environments are excluded. Original paths and private provenance may occur inside checkpoints/records; this is a recovery backup, not the anonymous reviewer packet or final combined team submission.
