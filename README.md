# DATA266 Lab 1 — Pair 49

Shared repository for Srinidhi ([Mrnidhi](https://github.com/Mrnidhi)) and
Revanth ([Revanth0211](https://github.com/Revanth0211)). Each member's code,
checkpoints, outputs, metrics and analysis live under their own folder in each task.

## Results and files

| Task | Srinidhi | Revanth |
|---|---|---|
| Character GPT | [Results](task1_llm/srinidhi/results.md) · [notebook](task1_llm/srinidhi/src/gpt.ipynb) · [metrics](task1_llm/srinidhi/metrics_report.csv): validation CE 0.551181, character accuracy 82.44% at context 512 | [Results](task1_llm/revanth0211/results.md) · [notebook](task1_llm/revanth0211/src/Part1.ipynb) · [metrics](task1_llm/revanth0211/metrics_report.csv): validation CE 0.622993, character accuracy 80.51% at context 256 |
| Yelp sentiment | [Results](task2_sentiment/srinidhi/results.md) · [notebook](task2_sentiment/srinidhi/src/sentiment.ipynb) · [metrics](task2_sentiment/srinidhi/metrics_report.csv): accuracy 93.31% MLP, 95.63% CNN, 96.12% BiLSTM | [Results](task2_sentiment/revanth0211/results.md) · [notebook](task2_sentiment/revanth0211/src/Part2.ipynb) · [metrics](task2_sentiment/revanth0211/metrics_report.csv): accuracy 92.845% mean baseline, 93.775% CNN, 93.130% BiGRU |
| CycleGAN | [Results](task3_gan/srinidhi/results.md) · [notebook](task3_gan/srinidhi/src/cyclegan.ipynb) · [metrics](task3_gan/srinidhi/full_metrics_report.csv): supplied-notebook composite 47.560426 | [Pending independent work](task3_gan/revanth0211/results.md) |

Data splits, contexts and evaluation protocols differ between members; these are
individual measurements, not a controlled model ranking. The current CycleGAN's
FID is 94.716036 and MiFID is 0.404816. Its local score is not a Kaggle score or rank.

```text
task1_llm/, task2_sentiment/, task3_gan/
  data/                         shared raw data
  srinidhi/, revanth0211/
    src/                        implementation and executed notebooks
    data_processed/             member-specific preprocessing, where applicable
    checkpoints/                trained weights and their manifest
    outputs/                    predictions, samples and plots
    metrics_report.csv          individual metrics (GAN also has full_metrics_report.csv)
    failure_analysis.md         failure/error analysis
    results.md                  model choices, results and hardware
reproducibility/
  manifests/                    environments, configurations and checkpoint/result mapping
  raw_logs/                     unchanged training evidence, including failed trials
  packages/                     large data/resume artifacts and portable submission packages
report/                         combined team report
```

The current GAN checkpoint and all 7,338 direct-output JPEGs are in Srinidhi's
member folder. Shared `src/lab1/`, `scripts/` and `tests/` provide execution,
packaging and checks. Personal paths, credentials, machine backups and temporary
working files do not belong in Git. Raw training logs are required evidence and
are retained; missing historical evidence is identified in the manifests.

## Setup and smoke test

```bash
git clone https://github.com/Mrnidhi/Data266_Lab.git
cd Data266_Lab
git lfs install
git lfs pull
bash scripts/smoke.sh
```

After cloning and fetching LFS artifacts, `bash scripts/smoke.sh` is the single
setup-and-smoke command on Linux/macOS. It prepares Python 3.12 dependencies and
checks the models on small synthetic CPU inputs. Outputs go to ignored `runs/`;
a successful smoke test checks execution, not trained model quality.

For an existing Python 3.12 environment, install `requirements.txt`,
`task3_gan/requirements.txt`, and the editable project:

```bash
python -m pip install -r requirements.txt -r task3_gan/requirements.txt
python -m pip install --no-deps -e .
```

GPU training needs a PyTorch/CUDA build supported by that GPU. Srinidhi's recorded
RTX 5090 runs used PyTorch 2.11.0+cu128; Revanth's RTX 4090 notebooks report
PyTorch 2.14.0+cu130. The root environment is the tested reproduction environment,
not a claim that both members trained with identical software.

## Reproduce each member's work

- Srinidhi: [Part 1 instructions](task1_llm/srinidhi/README.md),
  [Part 2 instructions](task2_sentiment/srinidhi/README.md), and
  [Part 3 instructions](task3_gan/srinidhi/results.md#reproduce-and-trace-the-result).
  These explain package preparation, saved-model notebooks and full-run recipes.
- Revanth: open [Part1.ipynb](task1_llm/revanth0211/src/Part1.ipynb) or
  [Part2.ipynb](task2_sentiment/revanth0211/src/Part2.ipynb) from its **src/** directory.
  For example, `cd task2_sentiment/revanth0211/src`, then
  `jupyter nbconvert --to notebook --execute Part2.ipynb --output Part2_executed.ipynb --ExecutePreprocessor.timeout=-1`.
  These notebooks download public datasets and train again; run in a separate
  clone because execution writes checkpoints and outputs. Their recorded parameters
  are embedded in the original notebooks; a new environment can change results.
- The [selected artifact index](reproducibility/manifests/selected.json) and
  [run manifests](reproducibility/manifests/) link results to saved checkpoints,
  environments, configurations and unedited logs. Exact training-source snapshots
  remain separate from later readability and resume-safety fixes.

## Contributions and assistance

Each member is responsible for their own three models, choices, analysis and
viva explanation. Commit under your own named folders and coordinate changes to
shared files. Do not overwrite another member's evidence or force-push history.

AI assistance was used for research, implementation, experiment operation,
debugging, packaging and drafted analysis. It is disclosed rather than presented
as independently authored student work. The brief requires the core decisions
and analysis to reflect the students' own understanding. Draft error analyses do
not count as completed student review, and AI ratings do not count as human ratings.

## Submission

The PDF requires `report/DATA266_Lab1_Report_Team_49.pdf`: one team ownership
paragraph, both members' architecture/hyperparameter/metric comparisons for all
three tasks, joint analysis, linked evidence, actual failure examples and the
three cited papers. The existing [report outline](report/REPORT_TEMPLATE.md)
is not that completed report.

Still outstanding: Revanth's Part 3; the combined report and joint analysis;
Srinidhi's student error reviews; a new blinded 30-sample GAN audit with two
human raters and agreement; and the team's
Kaggle submission and recorded rank. Some historical hardware/log evidence was
not captured and is marked missing rather than reconstructed as a measurement.
Earlier checkpoint ratings cannot be used for the current GAN.

Canvas requests one ZIP with Part 1, Part 2, Part 3 and a combined Report.pdf
including the repository link. Each part must contain executed notebooks,
weights and required outputs. Git publication is not Canvas or Kaggle submission.
