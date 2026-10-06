# Processed data

Raw Monet and photo images are not duplicated in this member folder. The run
validated 300 Monet JPEGs and 7,038 photo JPEGs before training. Images were
loaded as RGB, resized to 286, randomly cropped to 256 × 256, and randomly
flipped during training. Inference used deterministic 256 × 256 images.

The data identity used by the selected run is recorded in
`../outputs/selected_epoch_190/metrics.json` and
`../outputs/selected_epoch_190/inference_manifest.json`:

- data SHA-256: `6bbbc07362e75ff78ade98dec020b2a7a01ec4a892a8c5effa7010d2f6a578d0`
- reference-statistics SHA-256: `43c1616e28b6a8c5c5bdc91e1ece0fda6e07050f1c5e2b52c332090f411c18e0`

The 300-image deterministic checkpoint subset is listed in
`../outputs/selection_manifest.json`. Raw course data should be restored under
the task-level `task3_gan/data/` directory before reproduction.
