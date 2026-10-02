# Lab Pair 49 — compute allocation and portable runs

Updated October 1, 2026. The Parts 1/2 baseline and quality-study training completed on the user's Windows desktop. Cloud cost records below describe earlier Parts 2/3 work. This plan does not allocate resources.

The user requested Git consolidation first, then improvements to Parts 1/2.
The baseline technical snapshot was pushed to `main` as `87f00dc`, and only
`main` remains. A normal push of verified final study artifacts is authorized;
preserve existing history and do not force push. Complete raw evidence and
data remain in this workspace and verified archives.

## Current allocation

The user requested a fresh desktop Part 1 run after the earlier checkpoint download could not be located, and explicitly requested deletion of previous Part 1 runs. Their old cloud metrics and batch benchmarks no longer define the Part 1 publication. Parts 2/3 evidence is preserved.

| Work | Placement and status |
|---|---|
| Part 1: character GPT | Selected 6-layer/context-512 GPT completed 16 epochs and 28,416 updates on 100K/10K TinyStories. Native validation CE 0.551181; matched-256 CE 0.599380 versus baseline 0.721101. Evidence: `reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part1/depth_context_full/`. |
| Part 2: Yelp sentiment | Four further candidates completed on the desktop. Validation selection chose the wider MLP and retained the original BiLSTM/CNN. Selected epochs 5/11/6 from 9/12/6 completed epochs; test accuracy 93.3105%/96.1211%/95.6263%. The eight-cell notebook, CPU/CUDA inference and extracted-package notebook/CPU smoke checks passed. |
| Part 3: CycleGAN | Original 30-epoch run and two September 29 continuation arms completed on cloud RTX 5090. Selected continuation metrics and remaining review/class evaluation are in `report/PART3_FINETUNING_RESULTS.md`. |
| Reports, notebooks, packaging, checksums | Parts 1/2 technically finalized on Windows with actual weights/data in their verified ZIPs. Part 2 student review is pending. Part 3 large artifacts must still be transferred/verified if absent here; teammate comparisons and combined report are separate. |

The completed quality study is under
`reproducibility/raw_logs/srinidhi/desktop-quality-20261001/`. Promotion follows
the frozen validation rules. GPT and sentiment shared the RTX 5090 for part
of training, so elapsed times include contention and do not establish isolated
architecture-speed comparisons. Baseline ZIPs and raw evidence remain preserved.

The desktop environment is Python 3.12.14, PyTorch 2.11.0+cu128 and CUDA 12.8
with verified RTX 5090 support. See `verification/part1_desktop_environment.json`.
The selected GPT recorded 6,689.37 seconds of training-step work, 7,071.67
seconds overall, and 214,249.85 scored character/EOS targets per second during
timed training. The finalization receipts record notebook, inference and
package checks. No new GPU rental was used; each task retains its actual
hardware, environment, source and training provenance.

Part 2 uses the same pinned desktop environment with exact CPU
provenance in `verification/part2_desktop_environment.json`. New validation
selection preceded fresh test inference; old official-test results had
already been observed. The initial three-recipe replication and later four
candidate study have separate preserved plans. Measured training time for
the final selected MLP/BiLSTM/CNN is 457.4096/2,911.4808/240.7418
seconds for MLP/BiLSTM/CNN, excluding setup/validation/export; invocation
provenance records broader elapsed time. No new GPU rental was used.
See `PART2_FINALIZATION.md` and the fresh source run under
`reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected/`.

## Historical Part 2 cloud cost and stopping policy

The earlier session was observed from 2026-09-19 01:32:44 UTC to 02:43:51 UTC (about 71 minutes). Estimated compute plus container cost: **$1.18**, before any tax; this is a duration-based estimate, not an invoice. The RunPod console confirmed **$0.00/hour** after stop. See `verification/runpod_part_b_session_20260918.json` and the 96-file backup receipt. Existing GPU training processes finished normally, but new CUDA contexts became unavailable late in that session; its final inference therefore ran on the local Apple M3 CPU. No extra paid resource was started.

For that earlier Part 2 session, the user removed the initial $3 allowance and authorized continued research-guided experiments. The quoted rate at that time was $0.99/hour plus approximately $0.004/hour for a 30 GB container disk, before any tax. A real-data benchmark is in verification/sentiment_5090_benchmark.json. This historic authorization and price record does not call for a new rental; current Part 1 uses the desktop. Candidate runtimes depend on actual architecture, stopping epoch and workload.

Use a single GPU initially. The user authorized parallel independent work; separate validation-only candidate processes can share the existing GPU if measured throughput improves. More selected GPUs do not accelerate one single-device model. Quote any additional GPU allocation before starting it.

A paused Python process still incurs pod charges. Before stopping, copy checkpoints, logs and outputs to the Mac and verify their hashes. RunPod container storage is erased when the pod stops. No persistent volume is needed for this bounded run; retain complete portable checkpoints locally and stop the pod after the experiments and verified backups finish. Confirm the console shows no running compute or storage charges.

Sources: [RunPod pricing](https://www.runpod.io/pricing), [billing and storage behavior](https://docs.runpod.io/pods/pricing).

## Measure time rather than infer it from GPU names

1. Prepare frozen data and evaluator weights on CPU, record the code revision, then run setup and a brief CUDA smoke check on the assigned device.
2. Time representative full-architecture training batches after warm-up. Synchronize CUDA at timing boundaries; include the real data loader. Measure validation separately. Rehearsal is for timing/correctness and is not a final trained model.
3. Compute `remaining batches × measured seconds/batch + validation + final evaluation + checkpoint/export time`. Add at least 25% scheduling headroom and data/setup time separately. Use peak VRAM, examples/second and total wall time to decide allocation.
4. Full sentiment has 504,000 training rows: `ceil(504000 / 128) = 3,938` batches per epoch per model, at most six epochs in the reference and twelve in validation trials, with early stopping. The MLP, BiLSTM and CNN must be timed separately; early stopping may shorten training. Full GPT batches depend on the frozen stories' character-window count. The completed baseline GAN used `30 × 5,624 = 168,720` updates after frozen image splitting; the original class archive has 300 Monet and 7,038 photo files.

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

These commands require actual existing full-run checkpoints. The baseline snapshot on `main` includes Parts 1/2 selected member best weights. Both best/last files, full evidence and processed data are retained in `dist/Part1_Srinidhi_2342.zip` and `dist/Part2_Srinidhi_2342.zip`. Verified final study artifacts are authorized for a normal push to `main`; baseline evidence and archives stay preserved. For the selected wider MLP, run `git lfs install` and `git lfs pull` after cloning or pulling to obtain its complete 124,098,311-byte checkpoint. ZIPs retain full checkpoint bytes. Use the complete local archive/raw cache for preserved-selection evaluation, whose paths point to original source checkpoints. Check Part 3 availability against its own records. On Windows replace `.venv/bin/python` with `.venv\Scripts\python.exe`. Reapply original configuration overrides such as data locations. Preserve a run with logs beyond its saved checkpoint as evidence and resume into a fresh directory if the history guard rejects it. Validate one resumed batch and inference on the destination before starting a long session. Cross-device numerical equivalence requires a real hardware check.
