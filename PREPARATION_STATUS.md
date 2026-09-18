# Lab Pair 49 — preparation status

This is the shared team repository. Srinidhi's code and configurations live inside his named task folders; Revanth has separate `revanth0211/` scaffolds for his independent work. The initial commit contains prepared code and synthetic CPU verification, not final submission results.

## Verified locally

- 32 automated tests passed after moving the modules/configurations into the team layout.
- All three notebooks executed successfully: 15 code cells total, zero errors.
- All five model pipelines passed smoke training, evaluation and checkpoint checks.
- A relocated source archive passed import and smoke verification.
- New notebook outputs/raw logs use portable relative paths. Original earlier preparation logs remain unchanged in the separate local preparation folder.

Receipts are under `verification/`; raw runs are under `reproducibility/raw_logs/srinidhi/`. `bash scripts/smoke.sh` reproduces a fresh check from this source checkout. The Git repository excludes large datasets, model binaries, machine-specific environments, local metric-weight caches and preparation ZIPs.

## Pending

- Revanth must accept his GitHub invitation when sent and add his independent model contributions.
- An authorized GPU rehearsal: Linux/CUDA setup, GPU memory, throughput and time measurements.
- The working class Kaggle invitation/link and verified GAN data/rules; the PDF's link returned a missing-page message.
- Full training, all final metrics, real failure analyses, two human GAN raters, teammate comparisons and the combined report.

No RunPod was started by repository setup. The separate proposed rehearsal remains approximately $2 at the last observed RTX 4090 rate of $0.74/hour; see `RUNPOD_HANDOFF.md`. A Git push is not a Canvas or Kaggle submission.
