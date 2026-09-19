# Part 3 on Colab

The class data and A100 training pipeline are prepared. Full training and final
evaluation have **not** run yet. The next gate is the one-time Google consent for
the local Colab CLI to read, write and delete Colab resources. The isolated CLI
requests only basic identity and Colab scopes; it requests neither Google Cloud
project access nor Drive file access.

Use the existing Colab Pro for Education account with the previously verified
300 compute units. Check the account and remaining balance before allocation.
Do not buy credits or change its subscription. Request one A100, verify the actual
GPU, then record the actual compute-unit rate from Colab's Resources panel.

## Prepared experiment

- Two 9-block generators and two PatchGAN discriminators; RGB 256×256, batch 1.
- Adam learning rate 0.0001; 30 epochs, constant for 15 then linear decay for 15.
- BF16 forward passes with FP32 parameters, optimizer states and loss reductions.
- Seed 2342; 5,624/702/702 unique photos and 240/30/30 paintings across train,
  validation and test. All ten duplicate photo copies are excluded from splits.
- Select the checkpoint using validation KID every five epochs; test is reserved
  for final evaluation. The initial schedule contains 168,720 updates.
- All four models, optimizers, schedules, RNG states and replay pools are saved.

## Runtime setup and first measurement

The input ZIP contains allowlisted source files, real images and frozen splits;
it excludes credentials, local environments and unrelated course work. Its receipt
records per-file integrity verification and the complete archive SHA256.

```bash
# Local terminal, after authorized Colab-only login:
.colab-cli-venv/bin/colab whoami
.colab-cli-venv/bin/colab new -s data266-part3 --gpu A100
.colab-cli-venv/bin/colab upload -s data266-part3 dist/part3-colab-ready.zip /content/part3-colab-inputs.zip
.colab-cli-venv/bin/colab exec -s data266-part3 -f scripts/bootstrap_colab_part3.py --timeout 1800
```

Bootstrap verifies all input hashes, creates a separate pinned Python environment,
runs the CUDA-specific BF16/resume test, checks the metric implementations, and
records the actual environment. Active data and checkpoints stay on `/content`
for fast local I/O.

Run this inside the allocated runtime from `/content/Data266_Lab`:

```bash
.venv/bin/python scripts/run_part3_colab.py \
  --output reproducibility/raw_logs/srinidhi/colab-part3/full \
  --until-step 200
```

Read `segment_receipt.json`, the GPU utilization, and the actual Colab credit
consumption before extrapolating runtime. The receipt separates measured training
time from additional validation, data loading, backup and export time.

After the checks pass, resume the same recipe:

```bash
.venv/bin/python scripts/run_part3_colab.py \
  --output reproducibility/raw_logs/srinidhi/colab-part3/full \
  --resume reproducibility/raw_logs/srinidhi/colab-part3/full/last.pt \
  --until-step 5624
```

Increase the absolute `--until-step` for subsequent segments, or omit it to finish
the schedule. Make verified local checkpoint backups between segments. After an
interruption, preserve any log tail and resume into a fresh directory; retain the
previous `best.pt` beside `last.pt`. Do not modify frozen splits or precision.

## Completion

Evaluate the validation-selected checkpoint on both held-out directions. Export
predictions, figures, metrics, environment/data/code receipts, and a notebook with
actual outputs. Prepare blinded rating panels for two human raters. Exact class
CSV column names and the class scoring script still need to be verified; the
competition dialog currently specifies one data row plus a header. Local
Clean-FID results must not be relabeled as class leaderboard scores.

Download checkpoints and all evidence, verify hashes, then release the runtime:

```bash
.colab-cli-venv/bin/colab stop -s data266-part3
```

Do not run the final command until the required remote artifacts have verified
local copies. The final report must distinguish actual results from pending human
ratings, class leaderboard scoring and the teammate's independent work.
