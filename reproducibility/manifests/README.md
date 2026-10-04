# Reproducibility manifests

- `selected.json` maps current results to their checkpoints and packages.
- `training_evidence.json` indexes original logs, run configurations and environment records for the retained experiments, with file hashes and source locations. It also records missing evidence and original console files withheld because they contain personal paths.

Raw training files are preserved byte-for-byte under `../raw_logs/`. Notebook-output exports are explicitly labelled as later exports of stored execution output. They do not replace or rewrite the original notebooks.
