# Checkpoints

Selected validation-macro-F1 checkpoints from the completed RTX 4090 run:

| Model | Selected epoch | File | Size | SHA-256 |
|---|---:|---|---:|---|
| Mean-embedding baseline | 5 | `baseline_mean_best.pt` | 20,482,884 bytes | `5500aab3476feb4d008fe4b3f418313f483cee4a1f02cb63ea5cf39e7d2d2956` |
| Multi-kernel CNN | 5 | `cnn_multikernel_best.pt` | 26,836,108 bytes | `f33b528e8214a4e2579d9eb29fd22071911166d24b419ee48030eecb1dab729c` |
| Bidirectional GRU | 4 | `bigru_best.pt` | 26,495,163 bytes | `dff579aa7355f84e338fb4cbd2877d9989b90dc371ba724e7f2281463461270f` |

The digests and sizes are also machine-readable in
[manifest.json](manifest.json). Each checkpoint is paired with its model
definition and configuration in [../src/Part2.ipynb](../src/Part2.ipynb).
