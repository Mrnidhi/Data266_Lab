# Part 3 failure analysis

This is a preliminary technical review of the fixed examples in
`outputs/selected_epoch_190/translation_cycle_grid.png`. It is not a replacement
for the required blinded review by two independent people.

## 1. Source overlays are preserved

Photo `017018f377.jpg` contains a visible website watermark. The generated
Monet-style image still carries a faint version of the text. The model preserves
image content without understanding that the overlay is not part of the scene.

**Possible test:** remove or mask known text overlays during data preparation,
then compare the same fixed sample and FID/KID under an otherwise unchanged run.

## 2. Existing geometric distortion remains

Photo `01ae8be57e.jpg` already has strong perspective or panorama distortion
around the road and mountains. The translation retains the warped geometry and
adds painterly texture rather than repairing the structure. Cycle consistency
encourages the model to reconstruct the input, so it is not designed to correct
this kind of source defect.

**Possible test:** audit extreme-perspective images as a separate slice and
compare them with ordinary landscape photos. A crop policy that removes severe
panorama borders could be evaluated without changing the scoring subset.

## 3. Repeated texture and color cast

For photo `023637f8fb.jpg`, the translation applies a strong yellow-green cast
across the field and introduces repeated vertical texture in the sky. The broad
layout is preserved, but the repeated pattern can look artificial instead of
like natural brushwork.

**Possible test:** compare reflection-padded upsampling with an alternative
resize-convolution decoder and inspect the same fixed filenames for repeated
grid or banding patterns.

## 4. Monet-to-photo results remain soft

For Monet source `000c1e3bff.jpg`, the direct photo translation sharpens the
coastline and rocks but remains noticeably soft and painterly. This matches the
directional recall result of 0.176667: some generated features resemble photos,
but the output distribution covers a limited part of the photo domain.

**Possible test:** evaluate a lower identity weight or a longer discriminator
schedule while keeping the fixed checkpoint-selection subset unchanged.

## Scope

These observations use saved examples and suggest testable changes; they do not
prove the cause of each artifact. The empty two-rater sheet is preserved under
`outputs/selected_epoch_190/human_audit/ratings.csv`. Human ratings and agreement
must remain pending until both raters complete it independently.
