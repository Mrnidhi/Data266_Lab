# Outputs

`full/` contains the published full-run summaries, exact commands, configuration, data manifest, plots, held-out metrics, raw console log and fixed blinded audit panels. The complete per-image evaluation trees and 168,720-record JSONL log are retained in the verified local archive documented in `../checkpoints/README.md`; they are excluded from ordinary Git because of their size.

`pred_A2B/run-final/` contains 702 direct Photo-to-Monet PNG exports and `pred_B2A/run-final/` contains 30 direct Monet-to-Photo PNG exports. Each directory includes a checkpoint-hashed `export_manifest.json`. These are neutral inference exports with `submission_ready=false`; the class competition filename, archive and CSV rules must be verified before packaging a Kaggle submission.
