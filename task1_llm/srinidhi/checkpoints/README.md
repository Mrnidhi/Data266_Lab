# Part 1 checkpoints

The completed October 1, 2026 desktop RTX 5090 run wrote `best.pt` and `last.pt`
under `reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full/checkpoints/`.
Publication copied and hash-verified both files here after all twelve epochs
completed. The selected `best.pt` is prepared for Git publication with a targeted
ignore exception; `last.pt` remains a local
resumable checkpoint and is included in the Part 1 submission archive. The
published `manifest.json` records file sizes, SHA-256 hashes and original run
mapping. `../outputs/full/summary.json` records `best_checkpoint_epoch` and its
selected validation metrics.

Both files contain 27,183,983 bytes. The selected best checkpoint is epoch 12
at step 18,732, validation CE 0.7211012821932281. SHA-256 hashes:

- `best.pt`: `da90fb8bb2d7a71d9d33a75447a04c02345705307d9a9a39eff569468817e9b0`
- `last.pt`: `801304208fa196dac173d7b7b78dc6658817e3fcb0e1f1726b00d476d372f84d`

Both files contain model weights, optimizer/scheduler/AMP state, RNG state,
vocabulary, data manifest and training progress. Use the complete raw run and
both checkpoints when resuming; an isolated `best.pt` is suitable for inference
but does not provide the original logs and latest training progress.

The user requested replacement of previous Part 1 runs after the earlier
RunPod checkpoint download could not be located. No historic cloud checkpoint
or archive is claimed to be present. This directory's weights and hashes must
map to the new desktop run.

The final `dist/Part1_Srinidhi_2342.zip` includes both weights, executed notebook,
processed data, outputs and evidence. ZIPs are excluded from Git; keep an additional
verified copy outside this workstation for Canvas packaging. The selected
`best.pt` is the one exception to the repository's general model-binary ignore
rule. After the verified checkpoint is committed and pushed, cloning GitHub
can recover the trained Part 1 inference model. Check the remote commit before
assuming publication has occurred.
