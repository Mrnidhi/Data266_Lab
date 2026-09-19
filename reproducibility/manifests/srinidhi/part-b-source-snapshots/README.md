# Part B training source snapshots

These exact source bytes preserve the core versions used by the original suite, the first two validation-only candidates, and the later candidates/final evaluation. Match each run's recorded source hash to manifest.json. Shared source/configuration is available in Git commit 9105889; the complete runner and candidate recipes are committed separately.

The original suite kept its imported reference module for its entire process. Updating the source file for new candidate processes did not change an already running model. The added validation gate suppresses test evaluation during search. The later refactor extracts evaluation without changing the training updates.
