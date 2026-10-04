# Latest Part 3 — October 4, 2026

**Final supplied-notebook score: 47.56042586442388** (FID 94.71603584640337,
MiFID 0.4048158824443817). The below-44 target was not reached.

- [Part3.zip](Part3.zip): selected weights, all 7,338 unchanged predictions,
  original run source, executed results and scoring notebooks, metrics and logs.
- [Recovery.zip](Recovery.zip): six arm best/last checkpoints, portable warm-start,
  original matching code, settings and logs. Read its recovery caveats first.
- [Results notebook](Part3_Results.ipynb), [executed supplied scoring notebook](Part3_Evaluation_Script.executed.ipynb),
  [authoritative submission CSV](submission.csv), [verification receipt](receipt.json).
- [Existing dataset backup](../part3-20261002/dataset.zip), reused without duplication.

These archives preserve the original exported run. The current
[member folder](../../../task3_gan/srinidhi/results.md) also contains the later
required metric measurements and a notebook that runs directly from the repository.
The archives are unchanged evidence snapshots, not the complete team submission.

After cloning, run `git lfs install` and `git lfs pull`, then extract Part3.zip.
Open its `task3_gan/srinidhi/src/cyclegan.ipynb`; the standalone notebook displayed
here is for viewing and must be run with the extracted package's files.
The notebook completed two fresh CPU translations without errors. The full
unchanged evaluator already ran on the GPU server; packaging does not rescore it.
Both archives have verified CRCs and every member hash; see receipt.json.

The winner is batch-1 slow EMA at update 221250. Batch-1 completed; batch-8 and
R1 were interrupted by host OOM. All 192 scored candidates used class images
also seen during training, so this is not a held-out score. Historical packages
and original backups remain unchanged. Readable working code is a post-run
cleanup; the archives preserve exact source from the experiment.

See the member's results for current metric coverage. A new blinded human audit,
student review, the remaining teammate work/report and official Kaggle/Canvas
submission are not claimed. No old checkpoint's human ratings or metrics were reused.
