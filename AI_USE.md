# Lab Pair 49 — assistance and ownership

In the current Codex session, the user explicitly requested consolidating the
completed work on `main`, deleting merged extra branches, and then improving
Parts 1 and 2. The desktop baseline reached `main` at `87f00dc`; only `main`
remains locally and remotely. Further verified results may be committed and
pushed normally under that request. Existing history and raw evidence are
preserved. Earlier local-only instructions recorded by another session do not
describe this session's current authorization.

An AI coding assistant helped research candidate architectures, prepare code/configurations, build notebook entry points and run verification. On October 1, 2026 the user requested a fresh Part 1 run on the Windows desktop RTX 5090 and deletion of previous Part 1 runs after the older checkpoint download could not be located. The assistant prepared an isolated pinned environment, checked CUDA compatibility, fixed Windows/Unicode portability issues, operated the from-scratch run and prepared evidence-based publication/notebook/packaging tools. The new run's logs, manifests, weights and actual outputs establish its technical results. They do not establish that the student independently originated the core architecture or analysis. The original console remains local unedited; portable raw step/epoch metrics are preserved without rewriting them.

For the earlier Part B cloud work, the assistant prepared and benchmarked the full Yelp dataset, operated the authorized RTX 5090 reference suite and six validation-only comparison invocations (nine model candidates across seven training invocations), froze selection using validation scores, verified the local checkpoint backup and stopped the pod. Final evaluation and notebook verification ran on the Mac CPU. Those historical results are preserved separately from the new desktop reproduction.

On October 1 the user authorized continuing Part 2 on the Windows desktop. With the earlier weight/download files absent, the assistant restored the exact historical split rows and ordering, verified their hashes, prepared the encoded cache, fixed UTF-8 portability and host-memory/resume reporting, and implemented a frozen-plan runner and verification/package tooling. Architectures, tokenizer and training hyperparameters were unchanged. Three separately initialized validation-only desktop invocations completed the previously selected MLP/CNN six-epoch recipes and BiLSTM twelve-epoch/patience-four recipe on Intel Core Ultra 9 285K / RTX 5090. New validation selection froze epochs 5/11/6 and exact checkpoint hashes before fresh evaluation of all 38,000 official test rows. Accuracy was 93.1921%/96.1211%/95.6263% respectively. The assistant published measured metrics, executed all eight notebook cells with visible outputs, verified saved inference on CPU/CUDA and checked notebook/CPU smoke from the extracted package. Actual results are established by the new receipts. Historical official-test scores and error texts were already observed, so this is not a newly sealed test set. No fresh hyperparameter search was performed for that initial desktop reproduction; the later quality study is described below. The tuner reads test metadata/encoded arrays for integrity verification, then removes the test dataset; training receives only train/validation tensors, and no new test predictions or metrics inform selection.

The assistant prepared AI-assisted error-classification/explanation/testable-fix drafts for all twenty new cases from each of the three Part 2 models. These sixty model-specific entries are in separate `ai_error_review_draft.csv` files and explicitly labeled failure-analysis prose. Exact quote substrings, IDs, selected checkpoint hashes and full source-text hashes were checked; the original human packets and executed notebook remained byte-identical. All original `error_type`/`testable_fix` fields remain blank and every `student_reviewed` value remains false. Drafts do not establish manual student review. Hypotheses about model behavior or apparent label/text mismatch are not causal diagnoses or verified label defects. Proposed fixes require future training/validation studies and an appropriate new evaluation; no test-error-driven tuning, manual ratings or human review completion was fabricated.

For Part C, the assistant operated the completed 30-epoch class-data CycleGAN experiment on Vast.ai, prepared evaluation exports and blank human-review sheets, and helped run and report the September 29 continuation experiments. The selected continuation and its limitations are documented in `report/PART3_FINETUNING_RESULTS.md`. The user's personal review is recorded with explicit provenance; a second independent human review and inter-rater agreement remain pending.

For the subsequent October 1 quality study, the user requested research-guided
improvements to both parts. The assistant proposed and froze one larger
from-scratch GPT recipe and four validation-only sentiment recipes, with the
desktop baselines retained as controls. The GPT comparison uses matched
256-character validation windows even though its new training context is 512.
The sentiment selection rule gives a smaller model preference within 0.001
validation macro-F1. Optional ensembling/calibration and inference benchmarking
were removed from the submission scope when the user requested only the lab
PDF requirements. Calibration selection was stopped before a recipe or test
evaluation was produced; the local plan is preserved separately. Required Brier,
ECE, confidence intervals and paired comparisons remain part of the lab's
normal model evaluation. Part 2 selected the wider MLP and retained the
original BiLSTM and CNN under the frozen validation rule. Test accuracy was
93.3105%/96.1211%/95.6263%, respectively. Part 1 completed 16 epochs and
selected epoch 16. On matched 256-character windows, validation CE fell
from 0.721101 to 0.599380 (16.88%); evaluation-mode training CE was 0.591177,
with a validation-minus-training gap of 0.008204. Native 512-character
validation CE was 0.551181. Existing curves and these measurements were
used to discuss overfitting; no new test-driven adjustment was made.
These assisted experiment decisions and interpretations are
disclosed and do not establish independent student authorship or manual review.

The course brief requires the student's own core architecture decisions and analysis, independent models from each teammate and an individual viva defense. Assisted implementation, automated training and drafted interpretations do not themselves demonstrate those requirements. The student must review the code and actual outputs, verify interpretations, explain the choices and metrics, and address the course's assistance rules honestly. Technical completion of any part must not be described as complete authorship or whole-assignment compliance. Human ratings, teammate results and Kaggle scores must come from actual observations. No second human review, independent teammate result, official submission or student-review claim has been fabricated.

## October 2, 2026 — Part 3 assistance

At the user's request, the assistant researched and implemented a paired learning-rate/EMA continuation study, operated training and evaluation, checked checkpoint and output identities, drafted fixed-sample visual observations, and prepared the executed notebook and verified backup. Before any new endpoint validation result, the user's 11 AM lab-access cutoff prompted a documented amendment: compare the unchanged incumbent with epoch-four raw/EMA candidates in both learning-rate arms. The first arm completed eight epochs and the second stopped at four; epoch-eight candidates were excluded. Validation selected the LR `5e-5` epoch-four EMA model. Class and held-out results were reported after selection; earlier test/class results had already been observed.

The assistant prepared a fixed 30-sample packet with blank sheets for two human raters, repaired a notebook-only selection-path defect with preserved before/after evidence, and assisted the user-authorized Git backup. No current-model human ratings, official Kaggle upload/rank, independent student authorship or completed team submission are claimed. The [latest backup](reproducibility/packages/part3-20261002/README.md) retains original provenance; only its separate anonymous reviewer packet should be sent to raters. Student interpretation, two independent human ratings, teammate comparison and the combined report remain pending.
