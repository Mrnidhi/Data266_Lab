# Reproducibility evidence

Start with the verified packages:

- [Parts 1 and 2](packages/parts1-2-20261002/README.md): selected models, executed
  notebooks, data, metrics and reproduction instructions.
- [Current Part 3 — October 4](packages/part3-20261004/README.md): final outputs,
  the supplied evaluator result and separately preserved recovery states.

Large archives use Git LFS;
fetch their actual bytes before extracting them. Complete cloud-machine backups
and temporary experiments stay local under ignored `runs/`.

`raw_logs/` retains evidence required by the selected models; `manifests/` indexes
their artifact records. Superseded experiments stay out of the current tree. Preserve original
logs and source hashes; portable metadata copies record their original and
published hashes. `verification/` at the repository root holds execution receipts.

Use the same metric definitions and evaluation inputs when comparing teammates.
