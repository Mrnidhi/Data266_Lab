# Part 1 checkpoints

The selected model completed 16 epochs and 28,416 updates. Epoch 16 had the
lowest validation CE (0.551181 at context 512). The full training source is
`reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part1/depth_context_full/`.
The validation comparison also passed the frozen improvement rule at context 256.

Both `best.pt` and `last.pt` contain 69,230,751 bytes. The authoritative sizes,
hashes and source paths are in `manifest.json`:

- `best.pt`: `7945aa8a334f0b00c81aaf82a34137f41a34c4a7f2c1ef1986c2346b40c5bcc8`
- `last.pt`: `51a7919abd0a91baf964937476ff7d8d59c7e46f99df27596eed12ebb811faa2`

Each file contains model, optimizer, scheduler, AMP and RNG state, vocabulary,
data manifest and progress. Use `best.pt` for inference. Resume with `last.pt`,
the selected reproduction config and the complete matching raw run.

The selected `best.pt` is committed. The verified standalone package at
`reproducibility/packages/parts1-2-20261002/Part1_Srinidhi_2342.zip` contains both
states. From the repository root, run `git lfs pull`, then
`python scripts/prepare_srinidhi.py --part 1` to restore the exact `last.pt` and
frozen data before executing the notebook. Existing differing files are backed
up under ignored `runs/`; do not substitute an older run's last checkpoint.
