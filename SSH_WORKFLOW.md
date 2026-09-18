# Visible training over SSH

Status: SSH preparation only. The dedicated local key has not been registered with RunPod, the Pod is stopped, and no remote connection has been verified. The private key stays outside this repository on the user's Mac.

## Recommendation after inspecting RunPod

Use the existing single on-demand RTX 4090 for the initial bounded setup/benchmark session. Its current configuration exposes TCP port 22 and HTTP port 8888, with 16 vCPUs, 62 GB RAM and a 30 GB temporary container disk. Its stopped resume quote is $0.74/hour. The deployment page separately lists 30 GB container storage at $0.004/hour, so allow roughly $0.75 before tax for one powered-on hour.

The live GPU catalog also lists A5000 at $0.27/hour, 5090 at $0.99/hour and A100 at $1.59/hour. Those are hourly prices, not measured costs to finish this lab. A 5090 would need over 1.34 times the 4090's measured speed to reduce GPU compute cost; changing machines also adds setup/transfer overhead. Start with the expected college hardware and measure before considering alternatives.

For the first session, copy verified checkpoints to the Mac before stopping. The current container disk is erased on Stop. For repeated/full training, consider a network volume to preserve environment/data/checkpoints: standard pricing is $0.07/GB/month, so 50 GB is $3.50/month (roughly $0.12/day, prorated). It is a separate paid resource, persists while compute is stopped, and has not been created. Region availability and attaching/migrating to a compatible Pod must be checked first. Keep an additional local backup even with a volume.

Sources: [RunPod connection options](https://docs.runpod.io/pods/connect-to-a-pod), [Pod lifecycle](https://docs.runpod.io/pods/manage-pods), [storage billing](https://docs.runpod.io/pods/pricing). GPU prices and configuration were read in the signed-in console on September 18, 2026.

## Connect and watch

1. Register the dedicated public key after access confirmation; only its public half goes to RunPod. Start the existing Pod only after the session's cost/time is approved.
2. Copy the actual **SSH over exposed TCP** host and port from its Connect panel. Do not invent them or reuse an old port. Verify the first host-key fingerprint through a trusted connection; never bypass a changed-host-key warning.
3. In a local terminal, run the helper from the repository root. Replace the capitalized values with the verified connection details:

   ```bash
   bash scripts/connect_gpu.sh HOST PORT shell
   ```

4. On the Pod, verify `nvidia-smi`, free disk, and the environment. Install `tmux` if absent using the official OS package manager. Copy the prepared source bundle and data; do not place private GitHub/RunPod credentials in notebooks or logs. The private repository requires authorized authentication if cloning directly; the verified source ZIP avoids copying a GitHub credential to the Pod.
5. Reconnect using `bash scripts/connect_gpu.sh HOST PORT session`. It attaches to the persistent `data266` tmux session. Run setup/benchmarks there with unbuffered Python. Both the user and agent can attach to the same session. Detaching SSH leaves tmux training running; stopping/restarting the Pod does not preserve running processes.
6. A second local terminal can run `bash scripts/connect_gpu.sh HOST PORT gpu` to watch GPU utilization, VRAM and temperature every two seconds. The launcher retains SSH host-key checks and does not forward the local SSH agent.

All model runners now print task/model, epoch, batch/step, loss, session elapsed time and approximate remaining training-step time. Validation phases are announced. GPT/GAN use their configured logging intervals; sentiment defaults to every 50 batches and epoch-end. `eta_steps_est` is an estimate, not an end-to-end deadline; downloads, evaluation/export and early stopping change the total. Training elapsed time excludes time before training, so it is not the billing timer. The shared runner also saves terminal output to `RUN_LOG.txt`.

Use one trainer at a time initially. Measure speed/memory before full training or concurrency. Download and verify complete checkpoint bundles and logs before stopping the Pod. A training timeout, SSH disconnect, terminal close, or completed process does not stop billing. No automatic Pod shutdown or paid-resource allocation is implemented by these helpers.

## Validation boundary

The launcher has local syntax, argument and dry-run checks. Live model logging passes the existing CPU model/resume tests. These do not establish successful SSH connectivity, GPU memory fit, throughput or full model quality. The app terminal panel can be opened through the app tool, but direct computer-use control of Codex/macOS Terminal is restricted. Agent SSH execution can use command tools; the user can paste the same connection command into their terminal to watch the shared session.
