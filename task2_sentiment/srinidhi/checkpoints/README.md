# Selected Part 2 checkpoints

All six required checkpoint files are saved locally under the paths below. They were copied from the validation-selected source runs and verified by SHA-256. The executed notebook verifies all six hashes and reloads each best checkpoint for fresh CPU inference. Weights are excluded from ordinary Git.

| Model | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| maxpool_mlp | `maxpool_mlp/best.pt` | 62,361,671 | `fc9442519f52db62c292e380fed6285a14e51f2711e7540618314d7675720e08` |
| maxpool_mlp | `maxpool_mlp/last.pt` | 62,361,799 | `16a34b51fd4c4cb7e270e0479a4a30ecef9bb4de16666aeb8b65f956178d7421` |
| bilstm | `bilstm/best.pt` | 64,500,785 | `ed03e33534f053c0b12ceb519c719ea9716c4ece948c20715d6a535543003529` |
| bilstm | `bilstm/last.pt` | 64,500,977 | `5902cbd3c37ce0537b3389a16321824a7bc70373d4a2a114bbd9e9700781eba5` |
| dilated_cnn | `dilated_cnn/best.pt` | 67,308,375 | `013aa085d21b988169e37951027b280966e5dd550aa7769752e5daf657d0ed8a` |
| dilated_cnn | `dilated_cnn/last.pt` | 67,308,375 | `b0f85b1e7df44b7fdaacf6037519d3e9c2ad965a3599c74eed2a8069ad3fcb16` |

Portable local archive: `dist/part-b-srinidhi-20260918-checkpoints.zip` from the repository root. Size: 404,003,652 bytes. SHA-256: `ba845ee35266cf01a95be33570dcc8fb3fdecb8e341e434defef12c1228d31fe`. Archive creation verified every member checksum; it includes the derived full evaluation and selected best/last checkpoints, not the dataset or repository source. There is no hosted checkpoint download link yet. Include these actual weights in the final Canvas ZIP.

The complete original nine-model training backup is retained locally under `runs/part-b-full` and `runs/part-b-tuning`; its 96-file verification receipt is `verification/part_b_backup_verified.json`. Cloud container data were discarded only after this backup passed verification.
