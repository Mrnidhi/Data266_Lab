# Srinidhi — CycleGAN results

Status: actual full training and held-out evaluation are published. Human ratings, Kaggle results, student interpretation, and team comparison remain pending.

The model uses two 9-block generators and two PatchGAN discriminators, 64 base channels, 256px images, batch size 1, seed 2342, Adam learning rate 0.0001 and betas [0.5, 0.999]. The configured 30 epochs completed (168720 updates).

Cycle weight is 10.0 and the absolute identity weight is 5.0. These describe the recorded recipe; they do not by themselves establish that it is optimal.

Selected checkpoint SHA-256: `d8b37673ce3cbda3397740863fcde33126eb029a9cc5a7c7cdde9dcf064d9b7e`. Frozen data fingerprint: `35184a0643321f99efc8df06544328b25c3364b66f29c76545ab47e01fe3438a`.

| Direction | Metric | Value | Status |
|---|---|---:|---|
| photo_to_monet | fid | 195.82 | computed |
| photo_to_monet | kid | 0.0111121 | computed |
| photo_to_monet | precision | 0.60114 | computed |
| photo_to_monet | recall | 0.633333 | computed |
| photo_to_monet | density | 0.519658 | computed |
| photo_to_monet | coverage | 1 | computed |
| photo_to_monet | lpips_cycle | 0.188557 | computed |
| photo_to_monet | content_cosine | 0.810303 | computed |
| photo_to_monet | cycle_l1 | 0.0975051 | computed |
| monet_to_photo | fid | 188.238 | computed |
| monet_to_photo | kid | 0.019195 | computed |
| monet_to_photo | precision | 0.7 | computed |
| monet_to_photo | recall | 0.508547 | computed |
| monet_to_photo | density | 0.96 | computed |
| monet_to_photo | coverage | 0.143875 | computed |
| monet_to_photo | lpips_cycle | 0.236205 | computed |
| monet_to_photo | content_cosine | 0.901916 | computed |
| monet_to_photo | cycle_l1 | 0.121108 | computed |

Both CSV reports contain all recorded image-quality, training, parameter, hardware, and pending human/Kaggle fields. Unavailable values remain blank with a reason; pending values are not zero.

Local FID/KID use the recorded held-out evaluator; they are not verified class FID/MiFID or leaderboard scores. LPIPS compares the source with its own cycle reconstruction. Content cosine compares the source and its translation. Small Monet reference sets limit distributional metric precision.

The source training environment is in `outputs/full/run_summary.json`; the publisher's hardware is never substituted. Training throughput counts two source images per paired-domain update and excludes data loading/evaluation. Final-invocation wall time does not include earlier resumed sessions. Per-network losses and gradient norms retain their logged semantics.

If training resumed after a rollback, `outputs/full/history_reconstruction.json` records each explicit saved-update cutoff. Only derived curves and loss summaries exclude the discarded unsaved tail; original raw logs remain byte-identical.

Evidence: `outputs/full/raw_logs/`, `outputs/full/evaluation/metrics.json`, `outputs/full/figures/`, `outputs/full/resolved_config.json`, `outputs/full/data_manifest.json`, `checkpoints/selected.pt`, and `publication.json`. The executed `src/cyclegan.ipynb` reads these saved artifacts; its outputs do not imply notebook retraining.

## Exact recorded commands

```sh
.venv/bin/python -u scripts/run_part3_colab.py --output reproducibility/raw_logs/srinidhi/vast-part3/full --until-step 200
.venv/bin/python -u scripts/run_part3_colab.py --output reproducibility/raw_logs/srinidhi/vast-part3/full --resume reproducibility/raw_logs/srinidhi/vast-part3/full/last.pt
.venv/bin/python -u task3_gan/srinidhi/evaluate_local.py --mode full --device cuda --checkpoint reproducibility/raw_logs/srinidhi/vast-part3/full/best.pt --split test --output-dir reproducibility/raw_logs/srinidhi/vast-part3/evaluation-final --report reproducibility/raw_logs/srinidhi/vast-part3/evaluation-final/full_metrics_report.csv
```

## Analysis still requiring review

Inspect fixed source/translation/cycle examples for style changes, content loss, and artifacts. Explain observed convergence and stability using the actual curves, justify model choices in your own words, and compare with the teammate on the same evaluation protocol. Have two independent people complete the blinded audit; do not replace these ratings with automatically generated opinions.
