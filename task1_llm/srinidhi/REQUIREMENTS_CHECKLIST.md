# Part 1 requirement mapping — October 1 selected desktop model

Source requirements: DATA266 Lab1 Fall 2026 PDF, Task 1 on pages 6–7,
individual/team reproducibility and report instructions on pages 1–5, and the
provided Canvas submission message. This mapping concerns Srinidhi's Part 1.
Technical evidence does not establish student authorship or all-team completion.

The prior Part 1 runs were removed at the user's explicit request after their
checkpoint download could not be located. The completed baseline is preserved. The selected deeper model passed the
frozen context-256 validation comparison; its source is
`reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part1/depth_context_full/`.

## Task 1 technical requirements

The selected run completed all 16 epochs and 28,416 updates. The publisher
checked source hashes, complete epoch/target coverage, frozen-cache hashes,
checkpoint state, metric identities and exact failure excerpts; its receipt is
`verification/part1_export.json`. The finalizer passed all notebook, fresh
CPU/CUDA inference and archive inventory/CRC/SHA-256 checks. Seven notebook
code cells executed with seven visible output cells and zero errors. The raw
training inventory remained unchanged. The Part 1 package contains the
actual weights, executed notebook, frozen processed stories and required
outputs. See `verification/part1_notebook.json`,
`verification/part1_checkpoint_inference.json`,
`verification/part1_finalization.json` and
`dist/part1_package_verification.json` for technical finalization evidence.

| Requirement | Implementation / evidence | Status |
|---|---|---|
| Character-level TinyStories preprocessing | `src/gpt.py`, frozen member `data_processed/full/` JSONL and manifest | Verified |
| Own 100K training / 10K validation split | Seed 2342 indexed selection, source revision, row IDs and text hashes in `outputs/full/data_manifest.json` | Verified |
| Own integer mappings and fixed-length shifted sequences | `outputs/full/vocabulary.json` token order; notebook constructs `char_to_idx` / `idx_to_char`; story-bounded windows and padded-tail coverage | Verified; notebook executed |
| From-scratch multi-head attention, LayerNorm, feedforward, residuals | Manual Q/K/V attention and 6 pre-LayerNorm blocks in `src/gpt.py` | Implemented; causal test passed |
| Causal masking, learned token/position embeddings, LM head | Explicit lower-triangular mask and learned embeddings/head in `src/gpt.py` | Implemented; causal test passed |
| No prebuilt Transformer/attention modules or pretrained model | Member source; direct QK-transpose/softmax/V computation | Source reviewed |
| Cross-entropy, warm-up and learning-rate schedule | AdamW, 5% warm-up, cosine decay; config and raw step records | Verified |
| At least ten complete epochs | Sixteen completed epochs with no cap, every window/target counted; `history.json` and raw `metrics.jsonl` | Verified |
| Training and validation loss plots | `outputs/full/figures/learning_curves.png` and notebook | Published; notebook outputs visible |
| Greedy/temperature generation | Five fixed prompts, greedy and sampled continuations in `outputs/full/generations.json` | Verified and published |
| Three observed failure cases with actual snippets and type/observation | `failure_analysis.md` and executed notebook, each tied to a saved generation | AI-assisted draft; student review required |
| Every required evaluation metric | All required measures in the 38-row `metrics_report.csv`, with evidence/definitions in `results.md` | Verified and published |
| Hardware, environment, config and checkpoint mapping | `verification/part1_desktop_environment.json`, raw provenance and checkpoint manifest | Verified |
| Untouched raw training evidence | Raw `metrics.jsonl` steps/epochs and provenance; original console retained locally | Preserved |
| Executed notebook with all outputs visible | `src/gpt.ipynb`; `verification/part1_notebook.json` at repository root | Seven executed/output cells, zero errors |
| Saved actual model weights | `checkpoints/best.pt` prepared for Git publication; both best/last in Part 1 ZIP | Weights and SHA-256 verified |
| Required outputs and preprocessing available in Part 1 folder | `dist/Part1_Srinidhi_2342.zip`, folder `Part 1/`; `dist/part1_package_verification.json` | Inventory/CRC/SHA-256 verified |
| README can reproduce one smoke run using one documented command | Root/member README, native Windows CLI after pinned setup; `verification/part1_desktop_smoke.json` | CLI completed successfully in 2.33 seconds |

The metric CSV covers training/validation cross-entropy, perplexity,
bits-per-character, generalization gap, top-1 next-character accuracy,
Distinct-1/2/3, repeated 4-gram rate, gradient norms/NaNs/loss spikes, parameter
count, training/generation tokens per second, peak memory and training time.
Definitions distinguish online train loss from eval-mode validation loss,
character targets including EOS from word-token generation diversity, and
timed training-step work from end-to-end runtime. Empty diversity denominators
remain null. Undefined quantities are not replaced by invented scores.

For overfitting, the same-mode context-256 evaluation gives train CE 0.591177,
validation CE 0.599380 and gap +0.008204. Native validation loss improved at
every completed epoch; the minimum-loss epoch-16 checkpoint was retained.
The small positive gap does not establish performance outside the frozen split.

## Requirements still needing student/team work

| Requirement | Remaining action |
|---|---|
| Own core architecture decisions and analysis; permitted assistance | Student must address course AI-use rules honestly. Assisted code/results alone do not prove independent authorship. See the root README assistance section. |
| Review and individual demo/viva | Verify the code and actual failure snippets; explain design choices, metrics, masking, schedule and checkpoint selection. |
| Each member independently trains a distinct Part 1 model | Revanth's folder is a scaffold; obtain his actual independent code/config/weights/results. |
| Team comparison and joint synthesis | Compare every member's architecture, hyperparameters and full required metrics using the agreed protocol. |
| Combined final report with GitHub link and citations | Complete `report/DATA266_Lab1_Report_Team_49.pdf` across all three parts, with ownership statement and traceable evidence. |
| Final Canvas file layout | One ZIP with separate Part 1, Part 2, Part 3 folders and one combined `Report.pdf`; every part has executed notebook, weights and applicable outputs. |
| Outstanding Parts 2/3 obligations | Preserve their results; complete missing class evaluation, second human review/agreement and other checks in the repository submission checklist. |

No automated Part 1 check establishes that a teammate's work, student review,
viva, team report, Kaggle/Canvas submission or whole-lab completion occurred.
