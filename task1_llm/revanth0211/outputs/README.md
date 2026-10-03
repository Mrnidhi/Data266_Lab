# Outputs

Saved evidence from the completed run:

| File | Purpose |
|---|---|
| `metrics.json` | Machine-readable final metrics and resource measurements |
| `training_history.csv` | Epoch-level train loss, validation loss, accuracy, gradient norm, learning rate, and time |
| `training_log.txt` | Append-only console summaries; the final training-session block corresponds to the published metrics |
| `loss_curves.png` | Training and validation cross-entropy curves |
| `sample.txt` | Actual 500-character generated sample used by the failure analysis |

The concise submission copy of the final metrics is
[../metrics_report.csv](../metrics_report.csv).
