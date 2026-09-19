# Checkpoints

`best.pt` and `last.pt` from the completed RTX 4090 run are saved locally in this directory. Both contain all 12 completed epochs and 18,732 optimizer steps; the best validation checkpoint is epoch 12. Their byte counts and SHA-256 hashes are in `manifest.json`.

The complete portable run backup is `dist/part-a-srinidhi-20260918-checkpoints.zip` at the repository root (67,936,299 bytes). SHA-256: `8ee15fca517f0890ca790b3d751e11b65b2330520b18c6e97e02f5a9c2711023`. Its internal `run/` directory includes raw logs, data/source manifests, outputs and both checkpoints. Verify it with `python scripts/checkpoint_bundle.py verify --archive dist/part-a-srinidhi-20260918-checkpoints.zip` before moving or extracting it. Source code and the training data cache travel separately.

GPU inference reload, local tensor checks, source hashes and archive hashes passed. See `verification/part_a_completed_backup.json` and `verification/part_a_notebook.json`. The paid pod was stopped after verification.

Model binaries and ZIPs are excluded from ordinary Git; cloning the repository alone does not download them. No remote checkpoint-download location is published yet. Include the actual weights in the final Canvas ZIP and retain an additional backup outside the Mac before submission.
