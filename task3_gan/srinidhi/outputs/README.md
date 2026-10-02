# Outputs

**Current outputs are inside the [October 2 verified package](../../../reproducibility/packages/part3-20261002/README.md).** Extract the entire ZIP for the current notebook and its evidence. With the supplied class convention **A=Monet, B=Photo**, `pred_A2B/` contains 300 Monet-to-Photo JPEGs and `pred_B2A/` contains 7,038 Photo-to-Monet JPEGs. The current `submission.csv` and full metrics are linked from the package landing page.

`full/` and the records below are historical baseline evidence; their older A/B names used A=Photo and B=Monet. Current-model human ratings are pending. Give raters only the extracted `human_review_packet.zip` (30 fixed panels, 15 per direction); the complete backup retains original provenance and a private audit mapping.

## Historical baseline outputs

`full/` contains the published full-run summaries, exact commands, configuration, data manifest, plots, held-out metrics, raw console log and fixed blinded audit panels. The complete per-image evaluation trees and 168,720-record JSONL log are retained in the verified local archive documented in `../checkpoints/README.md`; they are excluded from ordinary Git because of their size.

`pred_A2B/run-final/` contains 702 direct Photo-to-Monet PNG exports and `pred_B2A/run-final/` contains 30 direct Monet-to-Photo PNG exports. Each directory includes a checkpoint-hashed `export_manifest.json`. These are neutral inference exports with `submission_ready=false`; the class competition filename, archive and CSV rules must be verified before packaging a Kaggle submission.
