# Part 3 class evaluation

The class convention is **A=Monet, B=Photo**. Translate every source image in
alphabetical order: 300 Monet-to-Photo JPEGs in `pred_A2B` and 7,038
Photo-to-Monet JPEGs in `pred_B2A`, each RGB 256×256.

The [supplied notebook](../reproducibility/packages/part3-20261002/Part3_Evaluation_Script.ipynb)
is the authority for the local class calculation. Its SHA-256 is
`702a1265433bf2f15c7900c83442c626d10ac0918094d093dde8ef82069d4cef`.

- The notebook sorts files and evaluates up to 300 per folder.
- Features use torchvision Inception-v3 with its supplied ImageNet preprocessing.
- FID compares real/generated feature distributions. The notebook's MiFID is
  mean cosine distance between real/generated features paired by sorted index.
  It is not interchangeable with other metrics bearing the same name.
- `submission.csv` has columns `ID,FID,MiFID`; ID is 1 and each metric is the
  average over both directions. The local composite is `(FID + MiFID) / 2`.

The [current executed notebook and CSV](../reproducibility/packages/part3-20261004/README.md)
report composite **47.56042586442388**. Selection used class reference images
also included in training. This score is not evidence of unseen-data quality,
and no Kaggle submission or rank is claimed.

Other metrics and human ratings must refer to this same selected checkpoint.
Unmeasured requirements stay marked missing; older models' measurements cannot
fill those rows. Generated images must remain authentic model outputs.
