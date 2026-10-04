PART 3 CYCLEGAN RUNNER
=====================

The package includes the code, class images, starting checkpoint, evaluation
notebook and Inception scoring weights. Install the Python dependencies before
running it. The runner does not upload results or manage GPU billing.

What it does
------------
1. Checks the environment and verifies package files against the SHA-256 manifest.
2. Runs the tests and measures training speed to set each experiment's schedule.
3. Trains the recipes in arms.json and saves their best scored checkpoints.
4. Exports direct generator outputs: 300 Monet-to-photo images in pred_A2B and
   7,038 photo-to-Monet images in pred_B2A.
5. Runs the unchanged class evaluation notebook and packages the results.

The --hours option limits the training stage. Scoring, export and packaging take
additional time. A longer run does not guarantee a better score.

Requirements and memory
-----------------------
Use Python 3.12 and a CUDA-compatible PyTorch build. The recorded RTX 5090 run used
PyTorch 2.11.0 with CUDA 12.8. The model recipes and defaults remain in arms.json.

Three concurrent experiments exhausted the recorded server's 32 GB of host RAM,
although GPU memory was available. Batch-8 and R1 were interrupted; batch-1 finished.
Check host RAM as well as GPU memory before repeating that plan. Reducing CPU
threads limits oversubscription but does not guarantee sufficient memory.

The approved CPU thread settings for the recorded run were all 4:
  OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS

Linux
-----
Extract the ZIP and enter the directory containing run_all.py. In a virtual
Python environment with CUDA support:

    python -m pip install -r requirements.txt
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Start a persistent terminal session on a remote server:

    tmux new -s part3
    export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
    python run_all.py --smoke

A smoke run is a rehearsal: 40 updates per experiment and batches capped at 2.
It does not establish full-run memory needs or run the complete scoring notebook.
Do not submit its SMOKE results ZIP.

After checking the rehearsal and available memory, start an authorized full run:

    python run_all.py --hours 10

Detach with Ctrl+B, then D. To watch progress from another terminal:

    tail -f work/run_all.log

Windows PowerShell
------------------
Extract the ZIP, open PowerShell and enter the directory containing run_all.py.
Use an existing compatible environment or create one:

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    $env:OPENBLAS_NUM_THREADS = $env:OMP_NUM_THREADS = $env:MKL_NUM_THREADS = $env:NUMEXPR_NUM_THREADS = "4"
    .\.venv\Scripts\python.exe run_all.py --smoke

Keep the terminal open for training. The runner asks Windows to remain awake.
Watch progress with: Get-Content work\run_all.log -Wait

Stopping and recovery
---------------------
Create work/STOP to request scoring, checkpointing and a clean stop. The runner
then exports the best saved candidate. Ctrl+C once also requests a clean stop;
after it exits, use --finalize-only to export without further training:

    python run_all.py --finalize-only

For an interrupted run, inspect work/arms/<name>.out and the saved checkpoints
before using --resume. Do not launch a second copy of an active experiment.

    python run_all.py --resume

Resume requires the original work/schedule.json when steps were automatic. This
source version fixes an older filename bug and refuses to replace a missing
schedule with a new benchmark. Historical packages remain unchanged: their
schedule may be named after the last experiment. Recover and verify the saved
schedule against checkpoint settings before attempting to resume such a package.
The time budget on a resumed invocation starts anew; it does not recover the
original wall-clock deadline automatically.

Automatic schedules require all configured experiments to run concurrently.
Passing --parallel 2 to the default three-experiment plan is not a supported
memory recovery shortcut. Review the experiment plan before a new run instead.

Results and interpretation
--------------------------
The final ZIP contains final/RESULTS.json, final/check.json, final/best.pt,
final/export/, the executed notebook, submission.csv and training logs.
Check that RESULTS.json has non-null official_notebook and checks entries.
A ZIP alone does not prove that notebook evaluation succeeded or all experiments
completed. Inspect each experiment's summary and exit messages too.

The class composite is (FID + MiFID) / 2; lower is better. The official score is
from the unchanged notebook, not the intermediate checkpoint-selection scorer.
Training uses all 300 Monets and 7,038 photos, including the scoring references.
Photos 301-600 were not used for checkpoint selection but were used in training;
the extra checks on them are not held-out generalization evidence. Report the
number of scored candidates because repeated selection makes the score optimistic.

check.json also contains KID, input/output feature cosine and nearest-neighbor
memorization diagnostics. These are diagnostics, not a substitute for visual
review or a guarantee of originality. Every exported image remains the direct
JPEG output of the selected generator, without editing, filtering or hand-picking.
Inception is used for scoring only. Kaggle submission is a separate manual step.

Download and verify the results before stopping a cloud instance. This script
never stops or destroys the instance; manage it through the provider separately.
