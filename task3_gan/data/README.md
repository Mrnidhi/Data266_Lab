# Shared image data

The class dataset contains 300 Monet and 7,038 Photo JPEGs. Restore the unchanged
images from the verified Git LFS archive after cloning. From the repository root:

```bash
git lfs pull --include="reproducibility/packages/part3-20261002/dataset.zip"
python scripts/prepare_srinidhi.py --part 3
```

The command verifies the archive SHA-256, each restored image and domain counts.
It writes only `task3_gan/data/monet_jpg/` and `photo_jpg/`, skips Mac metadata,
and preserves any differing local images under ignored `runs/` before replacing
them. Unexpected extra images cause an error rather than silently mixing datasets.

Raw images remain unchanged and are ignored by ordinary Git; their archive is
committed through Git LFS. Each member keeps split manifests and preprocessing
under their own `data_processed/`. This step does not alter any split or model.
