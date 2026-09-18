# Lab Pair 49 — RunPod handoff

No pod is started by any script in this package. The current preparation step does not authorize a paid run.

## Proposed rehearsal

Use one RTX 4090, matching the expected college hardware. The existing stopped pod was observed offering **$0.74/hour** on September 18, 2026. Recheck its displayed price before starting.

| Powered-on duration | GPU cost at $0.74/hour |
|---|---:|
| 30 minutes | $0.37 |
| 60 minutes | $0.74 |
| 90 minutes | $1.11 |
| 2 hours | $1.48 |

Allow roughly **$2 total** for a bounded preparation rehearsal, including some room for setup and storage. This is an estimate, not a provider-enforced cap. Startup, dependency installation, downloads, idle time, storage and taxes can add cost. Storage may remain billable while a pod is stopped. Full lab training cost/time is unknown until a real GPU benchmark measures the exact data and model.

Plan about 60–90 minutes of powered-on time for installation, CUDA tests, short training, checkpoint reload and artifact download, with a 2-hour operational limit. The training script's time limit only terminates training; **it does not stop the pod or billing**. A person must monitor the session, download outputs and stop the pod promptly.

## Once the user authorizes the start

1. Confirm the displayed GPU, hourly rate, storage and account credit. Start the existing suitable pod or one agreed replacement.
2. Clone https://github.com/Mrnidhi/Data266_Lab and open a terminal in that repository. Copy any prepared local data/metric caches separately, or regenerate them with the documented scripts.
3. Run `bash scripts/bootstrap_gpu.sh`. If the selected Python is not 3.12, it uses a small isolated uv installation to prepare Python 3.12 under this project; no system Python replacement is needed. Keep its output. Run `.venv/bin/python -m pytest -q` on the GPU machine. This Linux/CUDA installer remains unexecuted until pod authorization.
4. Run `.venv/bin/python scripts/rehearsal.py --device cuda --mode smoke --max-minutes 10` to catch CUDA-specific issues first.
5. Prepare/copy the text cache with `.venv/bin/python scripts/prefetch_rehearsal.py` if missing. Then run `.venv/bin/python scripts/rehearsal.py --device cuda --mode rehearsal --max-minutes 60`. It automatically uses the prepared cache if present. With no actual class image data, GAN results remain synthetic pipeline checks.
6. Check all run summaries, peak VRAM, steps/second, and checkpoint reload. If the 9-block GAN runs out of memory, explicitly reduce `base_channels` to 32, record that revised configuration, and rerun. Do not silently alter the model.
7. Download the entire `runs` and `verification` folders, including logs and weights. Verify files/checksums locally before removing anything remotely.
8. Stop the pod in RunPod and verify its stopped state. Inspect any remaining storage charges. Do not terminate/delete storage before confirming the download.

Use the observed throughput to estimate a full run: sum `(remaining training batches × seconds per batch) + validation + generation/evaluation + checkpoint overhead`, with at least 25% scheduling headroom. Short-run GPU timing must synchronize CUDA and exclude first-step initialization when forecasting steady state. Data loading and metric inference can be the bottleneck even on a larger GPU.

The optional `dist/lab1-evaluation-weights.zip` can be extracted beside the main ZIP, populating `lab1-2342/.cache/metric_weights`. It contains frozen evaluation networks, not lab model checkpoints. Verify copied weights with `.venv/bin/python scripts/prefetch_metric_weights.py --offline`. If omitted, download them with the same script without `--offline` before real image-quality evaluation. SciPy is pinned to 1.16.3 because Clean-FID's current implementation uses an API removed in SciPy 1.18.

Runtime setup follows [uv's documented Python installation support](https://docs.astral.sh/uv/guides/install-python/).

## College run

Confirm actual GPU/VRAM with `nvidia-smi`; the 4090 is expected, not verified. Copy the complete package and actual data, use the same tested versions, run the CUDA smoke check, then run full profiles. A cloud rehearsal does not establish that the college run occurred; retain its separate hardware log and outputs.
