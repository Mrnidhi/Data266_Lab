# Srinidhi — CycleGAN failure-analysis draft

This draft uses the completed full run and is intentionally separate from the required two-person blinded audit. The rater sheets remain blank, so no human score or agreement value is claimed here.

Training was numerically stable for all 168,720 updates: no NaN event was recorded. The cycle and identity losses fell sharply early and then declined more gradually, while the adversarial losses continued to fluctuate. This is expected for an adversarial objective and means the final generator loss alone should not be used to select the model. The checkpoint was selected by mean validation KID across both directions at update 168,720.

On the frozen test split, Photo-to-Monet had KID 0.01111, cycle L1 0.09751, LPIPS cycle 0.18856 and content cosine 0.81030. The fixed panels show clear color and brush-texture changes while major object boundaries are usually preserved. Some detailed foliage and water reflections become noisy or over-textured, which can obscure small structures.

Monet-to-Photo had KID 0.01920, cycle L1 0.12111, LPIPS cycle 0.23621 and content cosine 0.90192. Content is preserved strongly, but several outputs remain visibly painterly rather than fully photographic. Fine repeated textures can become harsh, and broad painted regions may retain brush patterns. Its held-out coverage is 0.14387, which is consistent with limited diversity relative to the much larger photo reference set.

The FID values (195.82 and 188.24) should be interpreted cautiously because the held-out domains are highly imbalanced: one direction compares 702 generated images with 30 real Monet references, and the other compares 30 generated images with 702 real photo references. The paired content and reconstruction measures, fixed panels and independent human ratings are therefore needed alongside FID/KID.

Remaining review: two people must independently score the 60 blinded panels for style, content and artifacts, after which the agreement report can be generated. The class Kaggle score and a comparison with Revanth's independently trained model are also pending.
