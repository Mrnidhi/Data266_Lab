# Srinidhi - Character GPT results

Completed 12 full epochs on NVIDIA GeForce RTX 4090 using 100,000 training and 10,000 validation stories. These are cloud GPU results; they do not claim a college-lab run.

The model has 3 blocks, 6 heads, width 192, context 256, dropout 0.15, batch 256, and 1,427,904 parameters. AdamW uses learning rate 0.0004, 5% warm-up and cosine decay. See README.md for the design and batch-size benchmark.

Best validation checkpoint: epoch 12. Validation cross-entropy: 0.719682; perplexity: 2.053780; next-character accuracy: 77.3761%.

All metrics are in metrics_report.csv. Character metrics include EOS targets. The generalization gap compares validation eval-mode loss with online training-mode loss in the selected epoch, rather than reevaluating the training split. Generation speed counts emitted characters, excluding the prompt and EOS. The loss-spike count uses successive logged steps and a 0.5 cross-entropy increase; it is a diagnostic, not proof of instability. The trainer stops on nonfinite losses or gradients.

Raw, unedited run evidence: ../../reproducibility/raw_logs/srinidhi/runpod-20260918/part-a-full-b256. Checkpoint checksums: checkpoints/manifest.json. The executed notebook displays the recorded GPU run; rendering it locally does not retrain the model. Failure analysis and interpretation should be reviewed by the student before submission.
