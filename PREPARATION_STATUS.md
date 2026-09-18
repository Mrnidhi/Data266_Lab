# Lab Pair 49 — preparation status

This is the shared team repository. Srinidhi's code and configurations live inside his named task folders; Revanth has separate `revanth0211/` scaffolds for his independent work. The initial commit contains prepared code and synthetic CPU verification, not final submission results.

## Verified locally

- 53 automated tests passed, including interrupted training, portable cache paths, scaler restoration, checkpoint transfer integrity and GAN best-checkpoint preservation.
- All three notebooks executed successfully: 15 code cells total, zero errors.
- All five model pipelines passed smoke training, evaluation and checkpoint checks.
- A relocated source archive passed import and smoke verification.
- All three synthetic CPU runs were bundled, copied, checksum-verified, extracted and resumed successfully. Sentiment now checkpoints within an epoch. CUDA device-count and FP16-scaler transitions have local mocked tests; actual cross-GPU migration is still pending.
- New notebook outputs/raw logs use portable relative paths. Original earlier preparation logs remain unchanged in the separate local preparation folder.

Receipts are under `verification/`, including `checkpoint_portability_pytest.xml` and `checkpoint_transfer.json`; raw runs are under `reproducibility/raw_logs/srinidhi/`. Earlier notebook and smoke receipts retain their original source provenance. `bash scripts/smoke.sh` reproduces a fresh check from this source checkout. The Git repository excludes large datasets, model binaries, machine-specific environments, local metric-weight caches and preparation ZIPs.

## Pending

- Revanth must accept his GitHub invitation when sent and add his independent model contributions.
- A GPU rehearsal on college/free resources, or RunPod after its expected cost is reviewed: Linux/CUDA setup, GPU memory, throughput and time measurements.
- The working class Kaggle invitation/link and verified GAN data/rules; the PDF's link returned a missing-page message.
- Full training, all final metrics, real failure analyses, two human GAN raters, teammate comparisons and the combined report.

No RunPod was started. The [compute plan](COMPUTE_PLAN.md) targets $0 additional spend by allocating independent tasks to college/free GPUs when available. Kaggle currently requires phone verification. Optional RunPod overflow remains approximately $2 for up to two hours at the observed $0.74/hour GPU rate, including a small storage allowance before tax; see `RUNPOD_HANDOFF.md`. Full training duration is unmeasured. A Git push is not a Canvas or Kaggle submission.
