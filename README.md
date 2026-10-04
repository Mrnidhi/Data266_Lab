# DATA266 Lab 1 — Pair 49

Independent implementations by [Srinidhi](https://github.com/Mrnidhi) and
[Revanth](https://github.com/Revanth0211): character-level text generation,
Yelp sentiment classification and unpaired Monet/photo translation.

## Latest results

| Part | Srinidhi's selected result | Files |
|---|---|---|
| 1 — Character GPT | Validation cross-entropy **0.5512**, perplexity **1.7353**, character accuracy **82.44%** at context 512 | [Results](task1_llm/srinidhi/results.md), [notebook](task1_llm/srinidhi/src/gpt.ipynb), [package](reproducibility/packages/parts1-2-20261002/README.md) |
| 2 — Yelp sentiment | Test accuracy: **96.12% BiLSTM**, **95.63% CNN**, **93.31% MLP** | [Results](task2_sentiment/srinidhi/results.md), [notebook](task2_sentiment/srinidhi/src/sentiment.ipynb), [package](reproducibility/packages/parts1-2-20261002/README.md) |
| 3 — CycleGAN | Supplied scoring notebook: **47.560426** composite, **94.716036** FID, **0.404816** MiFID | [October 4 package, notebook and results](reproducibility/packages/part3-20261004/README.md) |

Part 3's lower score is better; the target below 44 was not reached. The selected
batch-1 model uses slow EMA at update 221,250. Batch-1 completed 370,000 updates;
batch-8 and R1 experiments were interrupted by host-memory exhaustion. Selection
compared 192 scored candidates using class references also present in training,
so this is not a held-out estimate. No Kaggle upload or rank is claimed.

Revanth's independent Part 1 and Part 2 notebooks are in his member folders.
His Part 3 folder remains a scaffold. The combined report and remaining student
reviews are still pending; see [submission](#submission).

## Where to look

```text
task1_llm/           Character GPT
task2_sentiment/     Yelp sentiment models
task3_gan/           CycleGAN
  <member>/         src, checkpoints, outputs, metrics and analysis
src/lab1/           Shared training and evaluation entry points
scripts/            Reproduction, packaging and verification utilities
tests/              CPU checks and optional hardware checks
reproducibility/    Selected-run evidence and downloadable packages
verification/       Test and execution receipts
report/             Combined report outline
```

Each task keeps `srinidhi/` and `revanth0211/` separate. The latest Part 3 package
contains its selected checkpoint, all 7,338 direct-output JPEGs, executed
notebooks, official CSV and recorded training source. Its separate recovery ZIP
preserves saved training states. Superseded run outputs and planning notes are
excluded from the current tree.
Earlier human ratings do not describe the October 4 model.

Large required archives and the wider MLP checkpoint use Git LFS. Complete
machine backups, caches and temporary experiments stay local under ignored
`runs/`; they are not needed to browse the repository.

## Get the code and artifacts

```bash
git clone https://github.com/Mrnidhi/Data266_Lab.git
cd Data266_Lab
git lfs install
git lfs pull
```

For only the latest Part 3 result, fetch that folder and the shared dataset:

```bash
git lfs pull --include="reproducibility/packages/part3-20261004/*,reproducibility/packages/part3-20261002/dataset.zip"
```

Use Python 3.12 in a virtual environment. Install `requirements.txt` and the
editable project (`python -m pip install --no-deps -e .`). NVIDIA training needs
a PyTorch build compatible with the GPU; the recorded RTX 5090 run used
PyTorch 2.11.0 / torchvision 0.26.0 with CUDA 12.8. Additional image metrics use
`task3_gan/requirements.txt`.

On Linux/macOS, the preparation smoke test creates an environment and briefly
checks all five models on synthetic CPU inputs:

```bash
bash scripts/smoke.sh
```

On Windows, after environment setup, a Part 1 CPU check is:

```powershell
.venv\Scripts\python.exe -m lab1.run --task gpt --mode smoke --device cpu
```

Smoke outputs establish execution, not trained model quality. For real runs use
the task/package instructions and the recorded data and configurations. Training
scripts do not rent hardware or stop provider billing.

## Reproduce the results

- [Part 1](task1_llm/srinidhi/README.md): GPT setup, selected recipe and verification.
- [Part 2](task2_sentiment/srinidhi/README.md): sentiment setup, selected recipes and verification.
- [Part 3](reproducibility/packages/part3-20261004/README.md): selected outputs,
  exact trained source and checkpoint recovery. Readable working source includes
  later formatting and resume-safety fixes; archived run artifacts are unchanged.

## Contributions and assistance

Srinidhi and Revanth maintain separate member folders. Coordinate shared files
and comparisons; record which member produced each result. Use branches for
contributions and avoid force-pushing shared history.

AI assistance was used for research, implementation, experiment operation,
debugging, packaging and drafted analysis. Students must personally review and
explain their choices, code and results. AI-assisted error drafts are not human
ratings or completed student review.

## Submission

Canvas needs one ZIP containing **Part 1**, **Part 2**, **Part 3** and a combined
**Report.pdf** with the GitHub link. Each part needs executed notebooks, saved
weights and required outputs. Pushing to GitHub does not submit to Canvas or Kaggle.

Before submission, complete the student error reviews, missing metrics for the
selected GAN, two independent ratings and agreement, Revanth's Part 3 and the
team comparison/report. Earlier model measurements and reviews cannot be reused
for the current model. Use the [report outline](report/REPORT_TEMPLATE.md) and
the requirement checklist within each task. The supplied Canvas screenshot
shows October 6 at 6 PM; confirm the deadline in the course.
