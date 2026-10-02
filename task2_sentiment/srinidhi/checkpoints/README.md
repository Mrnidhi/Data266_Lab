# Fresh desktop Part 2 checkpoints

These six files belong to the October 1, 2026 Intel Core Ultra 9 285K /
RTX 5090 desktop run. They were selected by validation and copied from
recorded source runs; final metrics map to these exact best hashes.
The authoritative inventory is [manifest.json](manifest.json).

| Model | Best epoch | Best bytes | Best SHA-256 |
|---|---:|---:|---|
| MLP | 5 | 62,361,863 | `8e3007821edf5d69b3d485b7c3d5a534c2ad0b83021895e8ced26e4d187128b6` |
| BiLSTM | 11 | 64,500,977 | `0758cbe7405758fe6f99079f02fd3359aa1367d120cdd984c361c03a34fd54ba` |
| CNN | 6 | 67,308,503 | `f3883a005a369ef495c99bebe3cd51c4a7858c3f58121b6dbcfecee341dd8ac2` |

Each best/last checkpoint contains model/optimizer/scheduler/AMP state,
configuration, vocabulary, RNG/shuffle state and progress. Both files per
family are present locally and in `dist/Part2_Srinidhi_2342.zip`. The three
selected member `best.pt` files have targeted Git-ignore exceptions for Git
publication on `srinidhi/part2-desktop-final`. The final handoff and `git ls-remote`
establish the actual remote commit before a clone is used to restore them.
Resume `last.pt`, raw weight copies and dataset
caches remain ignored by Git and travel in the ZIP. No hosted ZIP/data
download link is claimed.

The eight-cell executed notebook checks all six hashes and reloads best
weights. Root `verification/part2_checkpoint_inference.json` records CPU/CUDA
finite-output, independent-reload and padding-invariance checks;
`verification/part2_package_portability.json` verifies real CPU inference
and notebook execution from the extracted archive. Current archive inventory,
CRC/file SHA and whole-ZIP SHA are in `dist/part2_package_verification.json`
and `dist/Part2_SHA256SUMS.txt`.

Earlier September cloud/Mac hashes and missing desktop download refer to
historical publication preserved under
`outputs/publication_history/b257a1079ef85973/`. They are not hashes or
availability claims for these new files. See [finalization guide](../../../PART2_FINALIZATION.md).
