# Reproducibility evidence

Raw logs and run manifests are grouped by member under raw_logs/. The initial Srinidhi runs are synthetic CPU smoke checks only. Preserve logs exactly as generated; new experiments use new directories. Model binaries are ignored by Git and must be backed up separately with checksum references.

The PDF's central `manifests/` directory contains byte-identical copies of the original run provenance, indexed by source path and SHA-256 under `manifests/srinidhi/index.json`. Copying them into the required location does not alter the originals.

Configuration and source hashes, package versions, data provenance and hardware disclosures are recorded with each run. Tests and notebook execution receipts are in the top-level verification/ directory. Do not edit old logs to make them look like new hardware or full training.

Agree on shared metric definitions and evaluation inputs before teammate comparisons.
