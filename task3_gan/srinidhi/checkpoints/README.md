# Part 3 checkpoints

**Current weights are in the [October 2 verified Git LFS package](../../../reproducibility/packages/part3-20261002/README.md).** After extracting the complete ZIP, use `task3_gan/srinidhi/checkpoints/selected.pt`: LR `5e-5`, epoch-four paired EMA, SHA-256 `0c2eab37c42062bbd7b3b2c5c5c097adc949535ff77a9f805ad89842f6f8073b`. This 113,182,935-byte candidate is for inference; it contains both EMA generators and same-update raw discriminators. Original warm-start weights and provenance are also in the archive. The current model's executed notebook and outputs travel with it.

## Historical baseline checkpoint records

The manifest and records below describe the earlier 30-epoch model. Their historical local archive paths are not download links; use the current package above for the latest complete backup.

The full 30-epoch run completed on Vast instance `51639909`. Model binaries are present locally and excluded from ordinary Git. Include both files in the final Canvas ZIP.

| File | Bytes | SHA-256 | Meaning |
| --- | ---: | --- | --- |
| `best.pt` / `selected.pt` | 378,941,587 | `d8b37673ce3cbda3397740863fcde33126eb029a9cc5a7c7cdde9dcf064d9b7e` | Validation-selected checkpoint at update 168,720 |
| `last.pt` | 378,941,587 | `34587ec68fc956e9d4a0b145e44698742aa1aad2833a945a93bd21304eb57719` | Final resumable checkpoint at update 168,720 |

Portable local run archive: `runs/part-c-vast-backup/vast-51639909/final/vast-part3-final.zip` from the repository root. Size: 3,167,791,674 bytes. SHA-256: `2b62217a5f66546803e100f00055aa67ac0227cf59357745ce62ccf3cb59fb41`. It contains 17,966 verified files, including both checkpoints, the complete 168,720-record training log, validation evidence and run provenance.

The final evaluation/publication/export archive is `runs/part-c-vast-backup/vast-51639909/final/vast-part3-final-artifacts.tar`. SHA-256: `f9ee9248e45a207c02005904f103d18727c9575cc531f5ae6db9abfd3615edaf`. There is no hosted checkpoint download link yet; the verified local copies are intended for the Canvas submission.
