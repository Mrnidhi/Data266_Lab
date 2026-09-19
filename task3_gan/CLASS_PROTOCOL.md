# Verified class competition protocol

Checked September 18, 2026 in the signed-in course and competition pages.

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
