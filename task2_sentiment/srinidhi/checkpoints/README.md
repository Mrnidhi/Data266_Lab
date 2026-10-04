# Selected Part 2 checkpoints

[manifest.json](manifest.json) maps current models to their exact best/last files. The wider MLP was selected from new validation candidates; BiLSTM and CNN retain their earlier desktop weights.

| Model | Selected / completed epoch | Best bytes | Best SHA-256 |
|---|---:|---:|---|
| MLP | 5 / 9 | 124,098,311 | `5ae7df4f747e89209ec2ca1381652bf3ed0fcf671a6f15605cb08323a0d9599c` |
| BiLSTM | 11 / 12 | 64,500,977 | `0758cbe7405758fe6f99079f02fd3359aa1367d120cdd984c361c03a34fd54ba` |
| CNN | 6 / 6 | 67,308,503 | `f3883a005a369ef495c99bebe3cd51c4a7858c3f58121b6dbcfecee341dd8ac2` |

Each checkpoint contains model, optimizer, scheduler and AMP state, configuration, vocabulary, RNG/shuffle state and progress. Use `best.pt` for reported predictions and `last.pt` to resume its training contract. The MLP uses embedding 256, hidden width 128 and dropout 0.5; it stopped after 9 of a maximum 12 epochs.

After cloning, run `git lfs pull` and
`python scripts/prepare_srinidhi.py --part 2` from the repository root. The
preparation command verifies the published package and restores matching
best/last states and frozen data. Existing differing files are backed up under
ignored `runs/`. The selected best files are committed; the larger MLP file
uses Git LFS, so a pointer must be replaced by its real contents before loading.

The original archive remains unchanged at
`reproducibility/packages/parts1-2-20261002/Part2_Srinidhi_2342.zip`. Notebook
inference verifies every checkpoint against this member's manifest. Use the
[member instructions](../README.md) for reproduction and evaluation.
