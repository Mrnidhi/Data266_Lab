# Part 1 outputs

`full/` contains the selected epoch-16 model's recorded summary, history,
greedy/sampled generations, loss curves, training diagnostics and vocabulary.
`reproduction_config.json` wraps the exact selected recipe for the runner;
the original member `config.json` still describes the baseline.

The canonical source is
`reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part1/depth_context_full/`.
Raw files are not rewritten. The original console stays local because library
warnings include host paths; the unchanged `metrics.jsonl` is portable evidence.

The executed `../src/gpt.ipynb` shows the actual recorded results and checks
saved-model inference. It does not train the model again. `../results.md`,
`../metrics_report.csv` and `../failure_analysis.md` explain the measurements
and three observed failures. AI-assisted interpretations still need student review.

The comparison receipt in `verification/part1_quality_comparison.json` uses
the same context-256 windows for both models and records eval-mode train and
validation losses. The context-512 native metrics remain separately labelled.
Baseline outputs are preserved under `reproducibility/baselines/` and in the
baseline ZIP; they are not recursively copied into this package.
