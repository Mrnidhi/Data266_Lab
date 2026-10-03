# Outputs

Completed evaluation artifacts from the three-model run:

| Artifact | Purpose |
|---|---|
| `metrics_report.csv` | Full model comparison; byte-identical to the required member-root report |
| `training_history.csv` | Epoch-level training and validation histories |
| `*_predictions.csv` | Test label, prediction, and positive-class probability for every model |
| `*_confusion_matrix.png` | Per-model confusion matrices |
| `roc_curves.png`, `pr_curves.png` | Comparative discrimination curves |
| `validation_loss_curves.png` | Validation-loss trajectories |
| `slice_metrics.csv` | Short, medium, long, and negation-slice macro-F1/error rates |
| `error_review_20.csv` | Completed 20-case manual review with proposed testable fixes |
| `class_distribution.png` | Label balance for all splits |
| `review_length_distribution.png` | Raw review-length distribution |
| `hardware_and_config.json` | GPU, library, seed, dataset, and shared settings |
| `vocab.json` | Training-only 40,000-token vocabulary |

The unedited compact console receipt is preserved as
[../RUN_LOG.txt](../RUN_LOG.txt).
