# Part 1 outputs

`full/` is the publication copied from the completed October 1, 2026 desktop
RTX 5090 run. It includes `summary.json`, `history.json`, `generations.json`,
`figures/learning_curves.png`, `figures/training_diagnostics.png` and
vocabulary/data metadata. Exact contents and
hashes are recorded by the publication and package verification manifests.
Final publication follows full training and saved-checkpoint verification.

The canonical raw run remains at
`reproducibility/raw_logs/srinidhi/desktop-20261001/part1-full/` at the repository
root. Its local original `RUN_LOG.txt` is not edited; library startup warnings
containing machine-specific paths make it unsuitable for Git. The portable raw
training log `metrics.jsonl` is tracked without rewriting it.

The executed notebook `../src/gpt.ipynb` renders actual recorded metrics,
curves, fixed-prompt generations and failure-analysis snippets. It documents
the original training command and loads the selected weights to verify
inference; rendering the notebook does not rerun twelve epochs. `../results.md`,
`../metrics_report.csv` and `../failure_analysis.md` tie claims to this run.
Student review of the generated-text interpretations remains required.
