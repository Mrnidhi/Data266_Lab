# Lab Pair 49 — compute allocation and portable runs

Checked September 18, 2026. This is an adaptive plan, not a reservation or a paid-resource authorization. The prepared models and dataset splits stay the same across platforms.

## Allocation

| Work | Preferred placement | Fallback and reason |
|---|---|---|
| Dataset preparation, packaging, plots and report | Local CPU or college CPU | Finish downloads/tokenization before renting a GPU. |
| Part 3: CycleGAN | College NVIDIA GPU, ideally the expected 24 GB RTX 4090 | Give the longest image job a stable session. Actual class images and split manifests are still missing, so do not start full GAN training yet. |
| Part 1: character GPT | Free Colab GPU while the college GPU runs GAN | Use the college GPU when free access is unavailable. Until GAN data are ready, GPT can use the college GPU immediately. |
| Part 2: all three Yelp models | A second available free platform, preferably Kaggle after verification | Otherwise run after GPT on Colab or on the college GPU. The current runner trains its three models sequentially and computes paired comparisons in one run. |
| Short CUDA rehearsal or deadline overflow | One RunPod RTX 4090, only if college/free resources are insufficient | Quote the actual GPU/storage rate and expected powered-on duration before starting. Do not start three paid Pods just to parallelize. |

Each platform runs an independent task, with a separate output directory. This does not require distributed training or networking between notebook runtimes. Do not run competing training jobs on a single GPU until a benchmark shows useful spare capacity. Extra GPUs in a Kaggle session do not automatically accelerate this single-device code.

## Availability observed

- **College:** GPU model, driver, permitted session length, free disk and access remain unverified. The 4090 is the user's expectation. Run `nvidia-smi` and a CUDA smoke check there. Assumed personal compute charge: $0 under normal course access.
- **Kaggle:** the signed-in notebook currently has GPU/internet features behind **phone verification**. It is not a usable GPU allocation yet. User verification, quota and a successful session must precede scheduling work there.
- **Colab:** free access is an option, but this task has not allocated or benchmarked a runtime. GPU model and uninterrupted runtime are not guaranteed. Use the notebook normally; do not build remote workers, bypass limits, or rely on a fixed quota.
- **RunPod:** the existing RTX 4090 Pod is **not running** and its console offers **$0.74/hour**, with a 30 GB container disk and no persistent volume shown. The console currently shows $0/hour. A stopped Pod is not a guarantee that the GPU will be available when resumed.

If free setup or queuing takes longer than about 10–15 minutes, continue useful work on the college machine rather than spending the session trying more providers. This threshold is a planning choice, not a platform limit.

## Cost before any paid start

The target incremental compute spend is **$0** using college access plus free notebooks. Keep RunPod as optional overflow. At the observed $0.74/hour GPU rate:

| Powered-on duration, including setup and idle time | GPU compute |
|---|---:|
| 30 minutes | $0.37 |
| 1 hour | $0.74 |
| 2 hours | $1.48 |
| 4 hours | $2.96 |

RunPod documents container storage at $0.10/GB/month while running; its 30 GB disk adds about $0.004/hour using a 730-hour month. Thus a two-hour contingency is approximately **$1.49 before tax**, or a **$2 planning allowance**. This is not a provider-enforced spending cap and is not a claim that all full training fits in two hours. Attached persistent storage can continue to cost money after compute stops. Recheck the console before starting; this estimate assumes the current configuration.

Stopping a Python script does not stop the Pod or its charges. Download and verify artifacts first, stop the Pod, and verify the console state. **RunPod erases the current container disk when the Pod stops**, so it must not hold the only copy of checkpoints. A larger GPU is worth renting only if its measured speedup exceeds its total hourly-price ratio and the task actually uses that capacity.

Sources: [RunPod GPU pricing](https://www.runpod.io/gpu-models/rtx-4090), [compute/storage billing and persistence](https://docs.runpod.io/pods/pricing), [Colab limits and VM lifetime](https://research.google.com/colaboratory/faq.html), [Kaggle notebook sessions](https://www.kaggle.com/docs/notebooks).

## Measure time rather than infer it from GPU names

1. Prepare frozen data and evaluator weights on CPU, record the code revision, then run setup and a brief CUDA smoke check on the assigned device.
2. Time representative full-architecture training batches after warm-up. Synchronize CUDA at timing boundaries; include the real data loader. Measure validation separately. Rehearsal is for timing/correctness and is not a final trained model.
3. Compute `remaining batches × measured seconds/batch + validation + final evaluation + checkpoint/export time`. Add at least 25% scheduling headroom and data/setup time separately. Use peak VRAM, examples/second and total wall time to decide allocation.
4. Full sentiment has 504,000 training rows: `ceil(504000 / 128) = 3,938` batches per epoch per model, at most six epochs each. The MLP, BiLSTM and CNN must be timed separately; early stopping may shorten training. Full GPT batches depend on the frozen stories' character-window count. GAN has `30 × max(training photos, training Monet images)` updates; the actual image counts are not known yet.

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

These commands require actual existing full-run checkpoints; none has been produced yet. Reapply any original configuration override (for example data locations). Preserve a run with logs beyond its saved checkpoint as evidence and resume into a fresh directory if the task's history guard rejects it. Validate one resumed batch and inference on the destination before committing a long session. Local CPU and mocked-device checks establish code behavior; real 4090-to-T4 migration still requires an actual GPU test.
