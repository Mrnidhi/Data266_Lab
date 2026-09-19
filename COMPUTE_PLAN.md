# Lab Pair 49 — compute allocation and portable runs

Checked September 18, 2026. This is an adaptive plan, not a reservation or a paid-resource authorization. The prepared models and dataset splits stay the same across platforms.

## Current allocation

Updated September 18, 2026 after the user reported that the college lab is closed for the next seven days. Complete training and submission preparation using the local Mac plus authorized cloud GPUs.

| Work | Placement and status |
|---|---|
| Part 1: character GPT | Completed on RunPod RTX 4090. Checkpoints, executed notebook, metrics and logs verified locally; pod stopped. |
| Part 2: Yelp sentiment | Completed the reference suite and six validation-only trials on one RunPod RTX 5090. All checkpoints and logs were verified locally before stopping; final inference and reporting use the Mac CPU. |
| Part 3: CycleGAN | Class archive received. Training and any GPU decision are deferred until Part 2 finishes and its pod stops. |
| Reports, notebooks, packaging, checksums | Local Mac, with verified local backups of every cloud training run. |

The earlier college/free-GPU allocation is superseded. No final submission may claim a college-GPU run that did not occur. Each task retains actual hardware, environment, source and training provenance.

## Part 2 cost and stopping policy

The completed session was observed from 2026-09-19 01:32:44 UTC to 02:43:51 UTC (about 71 minutes). Estimated compute plus container cost: **$1.18**, before any tax; this is a duration-based estimate, not an invoice. The RunPod console confirmed **$0.00/hour** after stop. See `verification/runpod_part_b_session_20260918.json` and the 96-file backup receipt. Existing GPU training processes finished normally, but new CUDA contexts became unavailable late in the session; final inference therefore runs on the local Apple M3 CPU. No extra paid resource was started.

The user explicitly removed the initial $3 allowance and authorized continued research-guided experiments for better Part 2 results. Keep one RTX 5090 active, measure throughput and costs, preserve checkpoints, and stop after useful work finishes. This is not authorization for unrelated purchases or an unbounded resource fleet. The current on-demand quote is $0.99/hour plus approximately $0.004/hour for a 30 GB container disk, before any applicable tax. A real-data benchmark is in verification/sentiment_5090_benchmark.json. Candidate runtimes depend on their actual architecture, stopping epoch and concurrent workload.

Use a single GPU initially. The user authorized parallel independent work; separate validation-only candidate processes can share the existing GPU if measured throughput improves. More selected GPUs do not accelerate one single-device model. Quote any additional GPU allocation before starting it.

A paused Python process still incurs pod charges. Before stopping, copy checkpoints, logs and outputs to the Mac and verify their hashes. RunPod container storage is erased when the pod stops. No persistent volume is needed for this bounded run; retain complete portable checkpoints locally and stop the pod after the experiments and verified backups finish. Confirm the console shows no running compute or storage charges.

Sources: [RunPod pricing](https://www.runpod.io/pricing), [billing and storage behavior](https://docs.runpod.io/pods/pricing).

## Measure time rather than infer it from GPU names

1. Prepare frozen data and evaluator weights on CPU, record the code revision, then run setup and a brief CUDA smoke check on the assigned device.
2. Time representative full-architecture training batches after warm-up. Synchronize CUDA at timing boundaries; include the real data loader. Measure validation separately. Rehearsal is for timing/correctness and is not a final trained model.
3. Compute `remaining batches × measured seconds/batch + validation + final evaluation + checkpoint/export time`. Add at least 25% scheduling headroom and data/setup time separately. Use peak VRAM, examples/second and total wall time to decide allocation.
4. Full sentiment has 504,000 training rows: `ceil(504000 / 128) = 3,938` batches per epoch per model, at most six epochs in the reference and twelve in validation trials, with early stopping. The MLP, BiLSTM and CNN must be timed separately; early stopping may shorten training. Full GPT batches depend on the frozen stories' character-window count. GAN has `30 × max(training photos, training Monet images)` updates; the class archive has 300 Monet and 7,038 photo files; image splits still need preparation.

No full-run hours or convergence guarantee is available before those measurements. Change GPU placement freely, but do not silently shrink the model, batch size, data or epoch schedule to meet a price estimate; changed training settings define a new experiment.

## Portable checkpoint contract

Use the same Git revision and pinned dependencies on every platform. Keep the selected data, IDs, text/image hashes, splits and vocabulary unchanged; move data/cache paths as needed. Train against local runtime storage for speed, and keep independently verified copies outside the runtime. Cross-GPU floating-point results may differ even when training state resumes correctly.

- GPT stores model, optimizer, scheduler, gradient scaler, RNG, vocabulary, split fingerprint and within-epoch position. Transfer the whole run so the previous `best.pt`, `last.pt`, logs and provenance remain together.
- Sentiment stores those states separately for each classifier, including shuffle state, best model and early-stopping history. The suite resume argument is its **run directory**, not a single model file.
- CycleGAN stores both generators/discriminators, both optimizers/schedules, replay pools, RNG and update position. Transfer the complete run plus the original image files and split manifests. A full resumable `last.pt` is roughly 400 MiB; allow space for both best/last checkpoints and outputs.
- Save every roughly 5–10 minutes of training after measuring throughput. The configured intervals are expressed in steps, not time. Tune checkpoint frequency without changing the training recipe. A local checkpoint alone does not survive runtime deletion.
- Transfer frozen text caches and GAN images/manifests separately from the run bundle. Synthetic rehearsal checkpoints cannot become genuine class-data checkpoints.

For planned moves, stop/pause the training process after a saved checkpoint, then make and verify a bundle with `scripts/checkpoint_bundle.py`. Its consistency checks reject files that change while being copied; it does not pause training for you. Keep two verified generations on independent storage, copy rather than move, and do not delete the source until destination verification succeeds. On Colab, copy finished archives to user-authorized Drive storage or download them. On Kaggle, explicitly save actual output artifacts or download them; session-local `/kaggle/working` by itself is not an off-session backup. No account storage upload or automatic synchronization has been configured here.

```bash
# Create on local runtime storage after pausing, then copy the archive off the runtime.
.venv/bin/python scripts/checkpoint_bundle.py create \
  --run-dir runs/gpt-full --output transfers/gpt-full-001.zip

# Run again after downloading/copying to the destination machine.
.venv/bin/python scripts/checkpoint_bundle.py verify \
  --archive transfers/gpt-full-001.zip
```

Extract a verified archive into a fresh directory. Its `run/` folder contains the complete original run; put that folder at the chosen output location. The helper verifies checksums without loading checkpoint pickle files. It does not upload, extract, synchronize, or stop any training/resource. Only load checkpoints from your own trusted runs.

After transferring code, run, and data, run from the cloned repository root and use the same mode/configuration:

```bash
# GPT: point at last.pt inside the transferred complete run.
.venv/bin/python -m lab1.run --task gpt --mode full --device cuda \
  --output runs/gpt-full --resume runs/gpt-full/checkpoints/last.pt

# Sentiment: resume all three models from the transferred suite directory.
.venv/bin/python -m lab1.run --task sentiment --mode full --device cuda \
  --output runs/sentiment-full --resume runs/sentiment-full

# GAN: first supply the verified, relocated image paths and split manifests.
.venv/bin/python -m lab1.run --task cyclegan --mode full --device cuda \
  --output runs/cyclegan-full --resume runs/cyclegan-full/last.pt
```

These commands require actual existing full-run checkpoints. Part 1 checkpoints are available; Part 2 and Part 3 completion must be checked against their own run records. Reapply any original configuration override (for example data locations). Preserve a run with logs beyond its saved checkpoint as evidence and resume into a fresh directory if the task's history guard rejects it. Validate one resumed batch and inference on the destination before committing a long session. Local CPU and mocked-device checks establish code behavior; real 4090-to-T4 migration still requires an actual GPU test.
