# Selected Part 2 checkpoints

[manifest.json](manifest.json) maps current models to their exact best/last files. The wider MLP was selected from new validation candidates; BiLSTM and CNN retain their earlier desktop weights.

| Model | Selected / completed epoch | Best bytes | Best SHA-256 |
|---|---:|---:|---|
| MLP | 5 / 9 | 124,098,311 | `5ae7df4f747e89209ec2ca1381652bf3ed0fcf671a6f15605cb08323a0d9599c` |
| BiLSTM | 11 / 12 | 64,500,977 | `0758cbe7405758fe6f99079f02fd3359aa1367d120cdd984c361c03a34fd54ba` |
| CNN | 6 / 6 | 67,308,503 | `f3883a005a369ef495c99bebe3cd51c4a7858c3f58121b6dbcfecee341dd8ac2` |

Each checkpoint contains model, optimizer, scheduler and AMP state, configuration, vocabulary, RNG/shuffle state and progress. Use `best.pt` for reported predictions and `last.pt` to resume its training contract. The MLP uses embedding 256, hidden width 128 and dropout 0.5; it stopped after 9 of a maximum 12 epochs.

Git LFS is configured only for `task2_sentiment/srinidhi/checkpoints/maxpool_mlp/best.pt`. After installing Git LFS, run these commands inside a clone after cloning or pulling:

```sh
git lfs install
git lfs pull
```

Verify the downloaded file against the manifest; an LFS pointer cannot be loaded as a model. Selected best files are intended for the authorized normal push to `main`. Raw copies, last checkpoints and data stay local and travel in the complete ZIP. The finalizer packages real full weights, including resume state.

Saved-model checks for these hashes passed and are recorded in `verification/part2_checkpoint_inference.json`. The extracted package's notebook and CPU smoke test also passed, recorded in `verification/part2_package_portability.json`. Use the final inventory and checksum in `dist/part2_package_verification.json` and `dist/Part2_SHA256SUMS.txt`.

The old desktop publication and baseline ZIP remain preserved separately. See the [finalization guide](../README.md) for current commands.
