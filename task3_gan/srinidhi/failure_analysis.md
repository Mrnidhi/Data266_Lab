# Failure analysis

This is a results-grounded draft for student review, not a completed human audit.

- The official score improved from the smoke run's 48.9991 to 47.5604.
  Smoke and final checks share class images, not an unseen test.
- Continuing batch-1 past its selected checkpoint did not improve the best saved
  score. Its final slow EMA interim score was 48.5550 versus 47.5443 at update
  221250. Checkpoint selection matters; the final training state is not the winner.
- Running three processes exhausted host RAM. Batch-8 and R1 failed with exit -9;
  finite logged losses do not mean these trials completed successfully.
- The fixed evaluation images were observed throughout selection. Lower local
  scores do not establish generalization. Future tuning needs a declared validation
  protocol and independent final evaluation.
- No checkpoint-specific human visual failure labels are asserted here. Inspect
  fixed blinded outputs, record two raters' scores and agreement, and explain
  actual content/style/artifact failures before final reporting.
