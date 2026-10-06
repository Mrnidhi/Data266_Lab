# Checkpoints

The selected run used the inference-only checkpoint `best_fid.pt` from epoch
190. The downloaded outputs reference that filename, but the checkpoint itself
was not included in either local Part 3 archive and is not available on this
computer.

Do not replace it with an older 128-pixel, 30-epoch, or epoch-200 checkpoint:
those files do not produce the published result. Download the exact
`runs/vast_256_seed3143_v1/checkpoints/best_fid.pt` file from the completed Vast
run, add it here through Git LFS, then record its SHA-256 and size in
`manifest.json`.
