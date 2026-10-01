# Verified class competition protocol

Initial check September 18, 2026; signed-in Kaggle recheck October 1, 2026.

## October 1 clarification and current state

- The [instructor's discussion reply](https://www.kaggle.com/competitions/data-266-fall-2026-gan-image-style-transfer/discussion/744502) says to translate **all 300 Monet images**, including images used in training, into `pred_A2B`. The class convention is **A = Monet, B = Photo**. This is the class export protocol, not evidence of performance on unseen images. Keep the independent validation/test manifests and their reports separate.
- Existing local neutral exports use the opposite A/B convention. For fresh class output folders, map `pred_A2B` to **`G_monet_to_photo`** and `pred_B2A` to **`G_photo_to_monet`**. Preserve the old outputs and their provenance; do not rename checkpoint keys or silently reinterpret historical metrics. Current numbered PNG exports are not class-ready JPEG exports.
- The current [leaderboard](https://www.kaggle.com/competitions/data-266-fall-2026-gan-image-style-transfer/leaderboard) lists 20 teams; Team 41 leads at **-41.4884**, followed by Team 38 at **-44.4076** and Team 05 at **-45.1953**. Team 49 is not listed. These are a dated snapshot, not a forecast.
- This account's Submissions page still shows **No submissions found**. No official score or rank is established for our model.
- The Overview still describes `(FID + MiFID) / 2`, lower being better, while the leaderboard ranks negative values with less-negative scores first. The exact sign/transformation must be verified against the official scorer; do not assume a formula from the displayed values.
- The Code tab still shows no notebooks. Data exposes the image folders and `real_stats.npz`, but no scoring script or sample CSV. Rules still specify self-reported FID/MiFID and a five-submission daily limit. The exact evaluator, direction aggregation, CSV schema, and full Photo export list remain unresolved.
- A focused local search also found no official evaluator/template. The supplied `real_stats.npz` has `mu_real` (2048), `sigma_real` (2048 x 2048), and `feats_real` (300 x 2048). These shapes alone do not establish which preprocessing, direction, or complete scoring rule the instructor uses.
- The known Canvas announcement currently redirects to SJSU sign-in. No credentials or permissions were changed. Obtain the instructor's evaluation script/template through the course account before claiming a comparable class score.

## Previously verified source and local preparation

- The [September 11 Canvas announcement](https://sjsu.instructure.com/courses/1629327/discussion_topics/5935481) supplies the updated private invitation. Access the invitation through Canvas; do not redistribute it.
- [Class competition](https://www.kaggle.com/competitions/data-266-fall-2026-gan-image-style-transfer): DATA 266 Fall 2026 - GAN Image Style Transfer. The account's rules page showed accepted status.
- The supplied class ZIP has 7,339 entries: 300 Monet and 7,038 Photo JPGs under `dataset/dataset/`, plus `real_stats.npz`. All images passed decoding checks as RGB 256×256. Ten duplicate photo copies were excluded from the frozen independent 80/10/10 splits; the original archive and all raw images remain unchanged. See `reproducibility/manifests/srinidhi/part-c-source-snapshot/` for hashes, duplicate mapping, and the six split lists.
- The domains are unpaired. Generated submission images must be 256 × 256 RGB JPGs. The existing direct-export PNGs need a separate documented JPEG export step.
- Rules describe `submission.csv` with self-reported FID and MiFID, at most five submissions per day. Exact CSV column names and row structure remain unverified.
- The Overview describes frozen Inception-v3 features and a class-specific MiFID proxy based on cosine distances after subsampling generated and real feature sets to equal sizes. Do not substitute a different memorization metric or assume that Clean-FID's preprocessing matches the class evaluator.
- The class score averages FID and MiFID. The page refers to a provided evaluation script, but the competition Code tab showed no notebooks. The script, feature extraction details, subsampling rule and CSV example still need to be obtained before creating class scores.
- The rules prohibit pretrained generators and manually edited/copied generation outputs. Frozen evaluation networks are allowed.

## Current blocker

The user supplied the class archive after the earlier browser-download failure. Data preparation is complete: Photo splits are 5,624/702/702 and Monet splits are 240/30/30 (train/validation/test), frozen at seed 2342. The archive contains no scoring script or sample CSV. The exact competition CSV schema and evaluator remain unresolved; training and separate held-out evaluation can proceed.

Local held-out Clean-FID/KID/PRDC/content/cycle metrics satisfy a separate evaluation purpose. They must not be relabeled as verified class leaderboard scores. Final Kaggle submission remains a reviewable, separate action.
