# Current Part 3 package — October 2, 2026

This directory backs up the completed, verified Part 3 work. **Use the notebook and model inside `Part3.zip` for the current results.** The older notebook and outputs in the repository's `task3_gan/srinidhi` folder remain historical evidence.

The immutable [Part3.zip](Part3.zip) contains the selected EMA checkpoint, its starting checkpoint, source, an executed notebook, all 7,338 class JPEGs, the supplied evaluator, class and separate held-out metrics, original study logs, plots, selection evidence, and a blank human-review packet. Both ZIP files use Git LFS. The [receipt](receipt.json) records their sizes and SHA-256 hashes.

| Class evaluator | FID | MiFID |
| --- | ---: | ---: |
| Monet to Photo | 98.1414649453 | 0.4185511172 |
| Photo to Monet | 101.7224012076 | 0.4071229398 |
| Directional mean | 99.9319330764 | 0.4128370285 |

The local arithmetic composite `(mean FID + mean MiFID) / 2` is **50.1723850525**, compared with 50.8594890609 for the starting model. FID improved in both directions; MiFID became slightly worse. The model was selected using the frozen five-candidate validation comparison, before this class evaluation. Mean validation KID fell by 10.75% to 0.0140057551. The deadline amendment compares epoch four in both learning-rate arms; it does not claim completion of the original eight-epoch study.

Current class labels are **A=Monet, B=Photo**: `pred_A2B` contains 300 Monet-to-Photo JPEGs and `pred_B2A` contains 7,038 Photo-to-Monet JPEGs. The supplied evaluator scores 300 predetermined images per direction. Older repository A/B filenames use a historical convention and must not be substituted.

- [Submission CSV](submission.csv), [class summary](class_summary.json), and [complete metrics](full_metrics_report.csv).
- [Unchanged supplied evaluator](Part3_Evaluation_Script.ipynb).
- [Original class dataset](dataset.zip), retained separately for full reproduction.

No official Kaggle upload, rank, top-five finish, 10/10 mark, or completed human audit is claimed.

For future fine-tuning, the [separate retraining backup](../part3-retuning-20261002/README.md)
preserves all checkpoint variants and both full optimizer/EMA/replay/RNG states.
Its guide distinguishes a new warm-start from continuing the original training state.

## Retrieve and open

After cloning this repository, run from its root with Git LFS installed:

```text
git lfs install
git lfs pull --include="reproducibility/packages/part3-20261002/*.zip"
```

On Windows PowerShell, extract to a fresh local directory:

```powershell
Expand-Archive -LiteralPath reproducibility/packages/part3-20261002/Part3.zip -DestinationPath runs/part3-from-git
```

On Linux/macOS:

```sh
unzip reproducibility/packages/part3-20261002/Part3.zip -d runs/part3-from-git
```

Open `runs/part3-from-git/Part3/MANUAL_STEPS.txt`, then `Part3_Results.ipynb` or the identical `task3_gan/srinidhi/src/cyclegan.ipynb` inside that extracted package. Its four code cells were executed with zero errors, including fresh CPU translations in both directions. The saved outputs can be read immediately; `RESULTS.md` provides the Python/dependency setup for rerunning. Running the notebook does not train a model. Full rescoring requires restoring the separately supplied dataset as described there.

`Part3.zip` is 1,230,602,460 bytes with SHA-256 `705cf7fc23d84aea8351a2acd9bbf256c4f7dd36df3000e4569ef318a9193f1c`. A tiny text file containing `oid sha256:` is an LFS pointer, not the downloaded archive. Use `git lfs pull`; do not assume GitHub's generic source-code ZIP includes LFS payloads.

## Remaining student and team work

This is a complete backup of the prepared Part 3 deliverable, **not the combined course submission**. The archive preserves original workstation paths and a private human-panel mapping for traceability. Do not present it as a path-clean team repository or share the whole archive with raters. Send only its separate `human_review_packet.zip` to the two independent raters: it has 30 anonymous panels, two blank sheets, and instructions. Complete genuine ratings, agreement, student review, teammate comparison, the combined report, and the official submissions separately. Earlier ratings belong to earlier checkpoints and cannot be transferred to this model.
