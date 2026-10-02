# Part 2 — AI-assisted error-analysis draft

This draft covers **twenty actual test errors for each of three fresh desktop models**
(60 model-specific entries). It does not claim manual student review. Each model has
five confident false positives, five confident false negatives, five near-threshold
errors and five prespecified long-review slice errors, with distinct IDs within its
packet. Reviews may overlap between models. These are deliberately selected cases,
not a random sample, so category counts do not estimate error prevalence.

Source: the October 1, 2026 Intel Core Ultra 9 285K / RTX 5090 run at
`reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/`. The separately
trained MLP/BiLSTM/CNN completed 6/12/6 epochs and selected epochs 5/11/6 using
validation before new test inference. Historical test scores/texts had already
been observed; this repeated test is not a newly sealed holdout.

The three `outputs/full/<model>/ai_error_review_draft.csv` files bind every draft
to the example ID, selected checkpoint SHA-256, unchanged human packet SHA-256
and full source-text hash. Quotes below are exact substrings of the human packet
text field. `verification/part2_ai_error_drafts.json` records ID/checkpoint/quote
checks and unchanged original packets/notebook. All sixty original human
`error_type`/`testable_fix` fields remain blank and `student_reviewed` remains
false. Students must inspect full texts and independently verify/correct each
interpretation before recording their own review.

Predicted probabilities and review text do not reveal a causal mechanism.
Mixed sentiment, target attribution, temporal updates, irony, input truncation
and rating/text mismatch are hypotheses. Negation is retained by preprocessing;
its removal is not an explanation. Long reviews retain their first 384 processed
tokens, but truncation alone need not explain an error. Official labels are kept
even when text appears inconsistent. Suggested fixes are future studies using
training/validation data and an appropriate new evaluation. They do not authorize
changing this test set, feeding these errors into training or tuning from them.

## Measured comparison and limits

| Model | Test accuracy | Macro-F1 | Brier | ECE (15 bins) | Incorrect / 38,000 |
|---|---:|---:|---:|---:|---:|
| maxpool_mlp | 93.1921% | 0.931917 | 0.052846 | 0.027214 | 2,587 |
| bilstm | 96.1211% | 0.961210 | 0.029803 | 0.014531 | 1,474 |
| dilated_cnn | 95.6263% | 0.956262 | 0.033397 | 0.012759 | 1,662 |

BiLSTM has the strongest single-seed accuracy and Brier score; CNN has the
lowest ECE, and the MLP is computationally simpler. Accuracy gains over MLP
are 2.93 percentage points for BiLSTM and 2.43 for CNN. Exact paired baseline
McNemar p-values are approximately 1.30e-124 and 1.55e-92 respectively (two
prespecified unadjusted comparisons). The bootstrap intervals concern test-row
sampling, not variability across training seeds. Seven raw-text hashes are
shared by training and test, so this is not a duplicate-clean benchmark.

The packets illustrate possible failure modes rather than proving the recurrent
model solves negation or the CNN fails on irony in general. All models share
the same word vocabulary, prefix cap and preprocessing. Candidate future studies
should change one factor at a time, predefine validation slices, preserve a
baseline and report aggregate performance, calibration and resource cost.

## maxpool_mlp

Selected best checkpoint: `8e3007821edf5d69b3d485b7c3d5a534c2ad0b83021895e8ced26e4d187128b6`.

### Confident false positives

#### 1. yelp:test:29330 — Apparent text-label ambiguity; positive surface language

Official label 0; predicted 1; P(positive) 0.999989; 23 processed tokens before the cap.

Evidence: “Wow love the place”; “Great place to come and relax worth a try!”.

**AI draft explanation:** The complete short review praises cleanliness, novelty and relaxation, with no explicit complaint. This makes the supplied negative label difficult to reconcile with the visible text and may explain a very confident positive prediction. It does not prove a labeling error or establish the model's causal features; the fixed test label must remain unchanged.

**One testable future fix:** Blindly audit a prespecified sample of short training and validation reviews for text-rating disagreement, then compare baseline training with a documented label-quality filtering rule fitted on training only. Select the rule and measure both macro-F1 and calibration on validation; do not relabel or tune using this test case.

#### 2. yelp:test:20681 — Underspecified two-word review; possible short-text overconfidence

Official label 0; predicted 1; P(positive) 0.999900; 2 processed tokens before the cap.

Evidence: “Large variety”.

**AI draft explanation:** The entire review is only a two-word description of selection. It could be approving or simply descriptive, and gives no explicit reason for its supplied negative label. The near-certain positive prediction may reflect learned lexical associations without enough context, but this is a hypothesis rather than an attribution result or proof that the label is wrong.

**One testable future fix:** Train a controlled MLP variant with label smoothing of 0.05 versus the original loss, using the same training split and seed. Compare short-review validation macro-F1, negative log-likelihood and confidence histograms before choosing a setting; retain all original test labels and reported predictions.

#### 3. yelp:test:25701 — Mixed sentiment with comparative and contrast qualifications

Official label 0; predicted 1; P(positive) 0.999880; 46 processed tokens before the cap.

Evidence: “good.....but not great”; “there are much better options”; “The service was very friendly”.

**AI draft explanation:** Food, service and cleanliness receive praise, while the central gordita judgment is qualified as merely good and inferior to alternatives. The model discards token order before classification, so it may fail to connect 'not' and 'but' to the praise they qualify. This plausible mechanism has not been demonstrated by ablation or attribution, and the mixed text does not justify changing its negative test label.

**One testable future fix:** Train a matched-parameter variant that applies width-3 contextual convolutions before pooling, alongside the original embedding-only max pool. Compare validation F1 on a prespecified contrast/negation slice and overall validation F1 before selecting the contextual representation.

#### 4. yelp:test:23815 — Uneven aspect coverage; possible text-rating ambiguity

Official label 0; predicted 1; P(positive) 0.999869; 129 processed tokens before the cap.

Evidence: “The bakery here is tops”; “The deli isn't the greatest but it will do in a pinch.”; “one of my favorite desserts of all time”.

**AI draft explanation:** Most of the review enthusiastically describes bakery products, while grocery selection is basic and the deli is only adequate. Positive bakery language may overwhelm the narrower limitations in a global pooled representation. However, the mostly favorable visible review also leaves the supplied negative rating ambiguous; neither an aspect-weighting mechanism nor a mislabeled example is proven.

**One testable future fix:** Build sentence-level encodings with learned attention over sentences, retaining the same training vocabulary. Compare them with global max pooling on a manually audited mixed-aspect training/validation slice, and choose the pooling method using validation results only; keep the fixed test label.

#### 5. yelp:test:12480 — Value complaint surrounded by food and service praise

Official label 0; predicted 1; P(positive) 0.999855; 13 processed tokens before the cap.

Evidence: “a little on the small side for $10”; “bacon was great”; “Our server was awesome!”.

**AI draft explanation:** The only explicit complaint concerns portion size relative to price; the omelette, bacon and server are praised. A max-pooled word representation may miss the multiword value comparison or give stronger weight to praise. The negative ground truth remains fixed, and this short mixed review could also reflect rating information not expressed clearly in the text; neither explanation is established.

**One testable future fix:** Add trainable width-3 and width-5 phrase features before pooling, keeping price tokens, and retrain on the original training split. Prespecify a validation value/portion-complaint slice and compare its F1 plus overall F1 against the original MLP before selecting the variant.

### Confident false negatives

#### 6. yelp:test:22807 — Brief review update followed by retained historical complaint

Official label 1; predicted 0; P(positive) 0.000041; 60 processed tokens before the cap.

Evidence: “EDIT: They really did change the service up since I last posted this.”; “Horrible service.”; “it doesn't give you any excuse to disrespect your paying customers”.

**AI draft explanation:** An opening edit reports changed service, but almost all remaining text recounts the earlier unpleasant encounter. The positive label may correspond to that later update. Order-insensitive pooling cannot explicitly distinguish an update from superseded complaints, making temporal attribution a plausible failure. The edit is itself brief and not unambiguously enthusiastic, so the text-label interpretation remains uncertain and no relabeling is warranted.

**One testable future fix:** Preserve sentence boundaries and an explicit EDIT marker in a future preprocessing variant, then train a sentence-order encoder on the unchanged training labels. Compare validation performance on a prespecified edited-review slice and the full validation set; do not alter this fixed test review or label.

#### 7. yelp:test:30958 — Target shift from corporate complaints to store-manager praise

Official label 1; predicted 0; P(positive) 0.000050; 235 processed tokens before the cap.

Evidence: “why do I give this store 5 stars? The Manager, Mario.”; “my opinion of Cox has changed”; “They are not the call center idiots.”.

**AI draft explanation:** The review distinguishes poor phone/corporate service from the Gilbert store manager who resolves the charges and earns five stars. All 235 preprocessed tokens fit within the 384-token input, so truncation does not explain this case. The order-invariant MLP may mix complaints about another service channel with praise for the rated store; this is a contextual hypothesis, not a demonstrated causal attribution.

**One testable future fix:** Train an ordered sentence encoder with attention pooling and compare it with the original max-pooled MLP using identical training/validation splits. Prespecify a validation subgroup containing entity or service-channel contrasts and assess subgroup plus overall F1 before choosing the representation.

#### 8. yelp:test:30793 — Old-owner complaints contrasted with current-owner approval

Official label 1; predicted 0; P(positive) 0.000091; 64 processed tokens before the cap.

Evidence: “when it was the old owners, it was terrible”; “Now its much better.”; “nothing but positive things to now say about this place”.

**AI draft explanation:** Both the opening and ending praise the new ownership; the negative incident is explicitly assigned to the old owners. The review is not truncated. The MLP's loss of word order makes distinguishing historical from current sentiment difficult, although it does not prove which words caused the negative prediction.

**One testable future fix:** Train a contextual width-5 convolutional feature layer before pooling, preserving words such as old, now and since. Evaluate against the original MLP on a prespecified temporal-contrast validation slice and overall validation F1, with architecture selection confined to validation.

#### 9. yelp:test:32713 — Food-shopping aspect conflict; uncertain aggregate rating

Official label 1; predicted 0; P(positive) 0.000106; 42 processed tokens before the cap.

Evidence: “Food:  Exceptionally nice”; “grossly overpriced”; “a little disappointed”.

**AI draft explanation:** The food is explicitly good, while the longer shopping discussion criticizes prices and local-produce availability. Negative shopping language may dominate the pooled representation even if food drives the positive overall label. Because the text gives no explicit overall rating, that weighting is uncertain and does not establish label noise or a model mechanism; the supplied positive test label remains fixed.

**One testable future fix:** Train a sentence-level attention variant that retains section markers such as Food and Shopping. Compare validation F1 on mixed-aspect reviews with the original model, and audit ambiguous examples only in training/validation before deciding whether aspect-aware pooling improves generalization.

#### 10. yelp:test:30354 — Qualified praise evaluated against different reference standards

Official label 1; predicted 0; P(positive) 0.000176; 66 processed tokens before the cap.

Evidence: “they are above-average for this area”; “this is a local 4”; “it's probably a 3”.

**AI draft explanation:** The author rates the restaurant four stars locally but only three relative to stronger restaurants elsewhere. The review also rejects exaggerated hype and calls some food merely average. Pooling without comparison scope may miss which reference standard supplies the positive label. This is a plausible explanation; the visible alternative ratings also make the sentiment boundary somewhat ambiguous, without permitting test relabeling.

**One testable future fix:** Train a contextual phrase encoder that preserves comparative clauses, then compare it with the baseline on a prespecified validation slice containing local-versus-external comparisons. Select the model using validation macro-F1 and retain the original test labels and existing results.

### Near-threshold errors

#### 11. yelp:test:903 — Near-boundary mixed review; food praise versus wait and value complaints

Official label 0; predicted 1; P(positive) 0.500145; 173 processed tokens before the cap.

Evidence: “the food took way too long”; “the burrito's are tasty and a good size”; “not my most favorable review”.

**AI draft explanation:** The burritos are praised, but the author repeatedly criticizes waiting time, price, atmosphere and the business's priorities. A score of 0.500145 is extremely close to the fixed decision threshold, consistent with competing evidence. The text supports a mixed-sentiment interpretation, but it does not prove why the pooled model placed slightly more weight on positive evidence.

**One testable future fix:** Compare global max pooling with concatenated max-and-mean pooling on the same training split, which may retain aggregate evidence as well as salient tokens. Prespecify mixed-aspect validation F1 and overall macro-F1 as selection criteria; any decision-threshold adjustment must be chosen on validation only.

#### 12. yelp:test:10370 — Near-boundary target praise surrounded by negative comparisons

Official label 1; predicted 0; P(positive) 0.499824; 242 processed tokens before the cap.

Evidence: “There are a few other lame run down outlets”; “This outlet is great for men!”; “the prices are awesome”.

**AI draft explanation:** Complaints largely concern other outlets, typical malls or shopping in general; the reviewed outlet is praised for selection and prices, with outdoor weather exposure its explicit downside. All 242 preprocessed tokens are retained. Losing order and target binding may mix irrelevant negative comparisons with target approval, although the near-threshold score alone cannot identify the cause.

**One testable future fix:** Train an ordered sentence encoder with a learned summary and compare it against embedding max pooling. Build a prespecified target-versus-competitor comparison subgroup from training/validation reviews and select the encoder using validation subgroup and overall F1, without retuning the fixed test prediction.

#### 13. yelp:test:20115 — Near-boundary qualified approval with negated criticism

Official label 1; predicted 0; P(positive) 0.498715; 206 processed tokens before the cap.

Evidence: “NOT nearly as bad as 99% of ethnic grocery stores”; “seriously this is not bad”; “for now I guess this place is a OK by me”.

**AI draft explanation:** The author acknowledges odor and limited variety but defends cleanliness, low prices and the store's overall acceptability. Expressions such as 'not bad' and 'not nearly as bad' qualify negative words rather than endorse criticism. The complete 206-token review is available to the model; missed negation/comparison scope is plausible but has not been established by a controlled intervention.

**One testable future fix:** Train width-3 and width-5 contextual convolution features before pooling, retaining negation words exactly as in the current preprocessing. Compare validation F1 on a prespecified negated-negative-phrase slice and overall F1 against the original MLP; choose all settings on validation.

#### 14. yelp:test:33180 — Near-boundary performance praise versus box-office pricing complaint

Official label 1; predicted 0; P(positive) 0.498092; 121 processed tokens before the cap.

Evidence: “It was flawless.”; “I would see it again.”; “at $70 less than mine”.

**AI draft explanation:** The acrobatics are praised and the author would return, while most of the later review criticizes misleading ticket-discount advice. 'Terrifying' describes daring performers rather than necessarily a bad show. A global representation may conflate excitement, ticketing dissatisfaction and performance quality; this is an aspect/lexical interpretation rather than proof of causal features.

**One testable future fix:** Compare an ordered sentence-attention model with global max pooling, using the same training vocabulary and splits. Prespecify a validation slice where the main experience and purchasing process receive different sentiment, and select the representation on subgroup plus overall validation F1.

#### 15. yelp:test:35551 — Near-boundary item-specific disappointment inside overall recommendation

Official label 1; predicted 0; P(positive) 0.497979; 100 processed tokens before the cap.

Evidence: “The ham and cheese and the El Capitan were very good”; “I was not impressed with the buffalo chicken empanada.”; “give this place a try, it is good”.

**AI draft explanation:** Two empanadas and the yuca are praised; criticism is restricted to the buffalo-chicken item, and the ending recommends the restaurant. Max pooling may preserve a strong negative item cue without representing how many aspects are positive or the final recommendation. That architectural limitation suggests a hypothesis, but no attribution experiment proves it caused this narrowly negative score.

**One testable future fix:** Train a max-plus-mean pooled MLP variant so repeated evidence contributes alongside extreme token features. Compare it with the baseline on a prespecified multi-item mixed-sentiment validation slice and overall macro-F1, choosing the variant and any threshold solely from validation.

### Long-review slice errors

#### 16. yelp:test:9911 — Long mixed hotel review with positive resolution beyond input cutoff

Official label 1; predicted 0; P(positive) 0.022513; 563 processed tokens before the cap.

Evidence: “A truly sincere gesture”; “These people really made our trip memorable”; “I'm giving this place 4 stars”.

**AI draft explanation:** The full review mixes hotel complaints with accommodating staff and an eventual room upgrade. Of 563 preprocessed tokens, only the first 384 are used; the final customer-service praise starts after 529 tokens and the four-star conclusion after 539. Earlier praise, including the returned phone, is retained, so truncation is a plausible contributing factor rather than a complete or proven explanation.

**One testable future fix:** Retrain the same MLP with a fixed 192-token head plus 192-token tail input, versus its original first-384 policy. Select the policy using overall validation F1 and a prespecified over-384-token validation slice, keeping vocabulary fitting training-only and leaving this reported test result unchanged.

#### 17. yelp:test:37318 — Long expectation-versus-experience contrast with omitted final verdict

Official label 0; predicted 1; P(positive) 0.669115; 561 processed tokens before the cap.

Evidence: “YU SHOuld not go to this place.”; “BOOOM!....AWAKEN FROM THE DREAM”; “The only thing elevated about this place is the price.”.

**AI draft explanation:** Early excitement describes imagined food, then the actual meal receives substantial criticism. Both the warning and several negative dish descriptions are inside the first 384 of 561 tokens, so the error cannot be attributed solely to missing negative text. The ending price verdict starts after 523 tokens and is omitted. Loss of chronological contrast and omission of the final summary are plausible contributors, not proven mechanisms.

**One testable future fix:** Run a controlled training/validation comparison of first-384 versus head-192/tail-192 inputs, then compare contextual width-5 features on the better validation policy. Prespecify long-review and expectation-versus-experience validation slices; do not use this test example to choose either change.

#### 18. yelp:test:9919 — Long qualified hotel approval; detailed service recovery truncated

Official label 1; predicted 0; P(positive) 0.065953; 536 processed tokens before the cap.

Evidence: “merited another star”; “600% my fault, had nothing to do with the escalator/hotel”; “THAT'S SERVICE.”.

**AI draft explanation:** The author considers the hotel workable despite many complaints and adds a star for kind, prompt treatment after an injury explicitly attributed to herself. The detailed incident begins after 396 of 536 tokens and is absent from the first-384 input, although the opening already states that good treatment merited another star. Missing resolution context may contribute, but retained positive cues prevent a truncation-only causal claim.

**One testable future fix:** Retrain using a fixed head-192/tail-192 selection with the same 384-token budget and compare against first-384 inputs. Choose the policy on a prespecified long-review validation slice plus overall F1, and separately examine validation service-recovery cases without changing any fixed test labels.

#### 19. yelp:test:25266 — Long initially enthusiastic stay reversed by hygiene and management failure

Official label 0; predicted 1; P(positive) 0.820363; 535 processed tokens before the cap.

Evidence: “HUGE COCKROACH IN BATHROOM OF OUR ROOM!!!!”; “Now is where the tables turned fast about this place.”; “He Chuckled.”.

**AI draft explanation:** The review opens with the hygiene complaint, then describes initial admiration before discovering the roach and receiving dismissive responses. Much of the manager's later response is beyond the first 384 of 535 tokens, but the headline and discovery are retained. The pooled representation may fail to distinguish initial expectations from the final experience, with truncation adding lost context; neither contribution is established causally.

**One testable future fix:** Train the baseline under first-384 and head-192/tail-192 policies, then compare an ordered contextual encoder if validation still shows expectation-reversal errors. Prespecify over-384-token and chronological-reversal validation slices and select changes using validation only, retaining the original negative test label.

#### 20. yelp:test:26354 — Long mixed meal review with decisive positive soup and rating omitted

Official label 1; predicted 0; P(positive) 0.137683; 524 processed tokens before the cap.

Evidence: “It was not dried out or overcooked, it was just not juicy.”; “That soup alone brought the 3-star lunch up to a 4-star lunch.”; “So I am holding at 4 stars.”.

**AI draft explanation:** The restaurant's setting and service are praised, followed by nuanced burger complaints and a soup that raises the overall rating. Of 524 tokens, the model retains only 384: the soup-based rating change begins after 445 tokens and the four-star conclusion after 503. The cutoff demonstrably removes important favorable context, but retained praise and untested model responses mean its causal contribution remains a hypothesis.

**One testable future fix:** Retrain the unchanged MLP using head-192/tail-192 inputs and compare with first-384 inputs at the same budget. Select the truncation policy from long-review and overall validation F1; additionally inspect mixed-meal validation cases before any future architecture change, without retuning or relabeling this test case.

## bilstm

Selected best checkpoint: `0758cbe7405758fe6f99079f02fd3359aa1367d120cdd984c361c03a34fd54ba`.

### Confident false positives

#### 1. yelp:test:29330 — apparent label/text mismatch

Official label 0; predicted 1; P(positive) 0.999996; 23 processed tokens before the cap.

Evidence: “Wow love the place”; “worth a try!”.

**AI draft explanation:** The visible review is uniformly complimentary despite the official negative label. The positive prediction is a scoring error under that label, but the text alone does not support diagnosing a sentiment-understanding failure. Rating/text mismatch or missing rating context is a hypothesis, not a verified label defect.

**One testable future fix:** Independently double-annotate potentially inconsistent training/validation reviews and measure agreement plus validation accuracy before and after a prespecified training-label audit; retain the official test label and full-test score unchanged.

#### 2. yelp:test:17407 — mixed sentiment and long-review context loss

Official label 0; predicted 1; P(positive) 0.999929; 453 processed tokens before the cap.

Evidence: “It was fantastic.”; “SOOOOOO TINY.”; “Bad place for someone who hasn't eaten”.

**AI draft explanation:** Past-visit praise and favorable dish descriptions coexist with dissatisfaction about the changed concept and small portions. The 453 processed tokens exceed the 384-token prefix cap, so the complete verdict is not necessarily represented. The very confident positive score does not establish which retained phrases dominated.

**One testable future fix:** Compare first-384 versus fixed head-and-tail token allocation using otherwise identical training and validation recipes, reporting validation macro-F1 and the predefined long-review slice.

#### 3. yelp:test:9706 — mixed aspect sentiment and conditional endorsement

Official label 0; predicted 1; P(positive) 0.999900; 164 processed tokens before the cap.

Evidence: “very beautiful”; “my feet are still hurting”; “if someone else was paying the bill”.

**AI draft explanation:** Praise for decor, rooms and staff contrasts with inconvenience, exhausting navigation and a qualified willingness to return. This 164-token review is not truncated; the error is consistent with failing to weight the negative overall value judgment, although pooling causality is unproven.

**One testable future fix:** Train a sentence-level aspect aggregation challenger and compare it with masked max pooling on a separately labeled validation subset containing conditional endorsements; keep the same training-only vocabulary.

#### 4. yelp:test:4655 — mixed sentiment with ironic crowd commentary

Official label 0; predicted 1; P(positive) 0.999888; 34 processed tokens before the cap.

Evidence: “good drink selection.”; “bad: the parking and the crowd.”; “totally the coolest place, like, ever.”.

**AI draft explanation:** The explicitly listed drawbacks and mocking description of the crowd coexist with several positive phrases. The positive prediction is consistent with literal reading of praise or weak aggregation of the good/bad structure; the intended degree of irony remains a human judgment.

**One testable future fix:** Build train/validation examples with explicit good/bad aspect lists and independently annotated irony, then test a sentence-attention head against the existing pooling head without using these test labels for training.

#### 5. yelp:test:5752 — temporal update versus historical praise

Official label 0; predicted 1; P(positive) 0.999834; 260 processed tokens before the cap.

Evidence: “This was a 4-star review”; “have gone down the tubes”; “All wonderful.”.

**AI draft explanation:** The opening note reverses an earlier four-star assessment, but most of the 260-token body praises the old experience. The strong positive output is consistent with discounting the update relative to historical praise. No truncation is needed to explain the conflicting evidence.

**One testable future fix:** Construct training-only old-review/update pairs and evaluate an update-aware sentence aggregation feature on held-out validation updates, reporting both overall and temporal-update macro-F1.

### Confident false negatives

#### 6. yelp:test:21215 — sentiment target attribution

Official label 1; predicted 0; P(positive) 0.000030; 100 processed tokens before the cap.

Evidence: “they completely took advantage of her”; “the bartender was very cordial”; “do the right thing”.

**AI draft explanation:** Negative language concerns the disruptive parents while the business's bartender is praised for handling them responsibly. The negative score is consistent with attributing criticism of patrons to the reviewed establishment; a target-aware interpretation supports the official positive label.

**One testable future fix:** Annotate sentiment target spans in an independent training/validation subset and compare a target-conditioned aggregation model with the baseline, measuring errors on third-party complaints separately.

#### 7. yelp:test:22807 — temporal edit and possible rating/text ambiguity

Official label 1; predicted 0; P(positive) 0.000042; 60 processed tokens before the cap.

Evidence: “They really did change the service up”; “Horrible service.”; “disrespect your paying customers”.

**AI draft explanation:** The edit implies a changed service experience, but the retained body is overwhelmingly negative and the edit does not explicitly say better. The official positive label may reflect a later rating. This is a temporal interpretation or label-context ambiguity hypothesis, not a proven mistaken label.

**One testable future fix:** Independently annotate update direction and overall sentiment in training/validation reviews, then test an edit-marker-aware representation against the original model while retaining official test labels.

#### 8. yelp:test:264 — target ambiguity and label/text tension

Official label 1; predicted 0; P(positive) 0.000089; 28 processed tokens before the cap.

Evidence: “I like Applebee's”; “something Yelp does”; “But not cool!”.

**AI draft explanation:** The reviewer likes the restaurant but complains about a delivery listing that may originate from Yelp or another editor. A negative literal overall reading is plausible, whereas the official positive label may emphasize the restaurant. Text alone cannot resolve the label's intended target.

**One testable future fix:** Double-annotate sentiment targets and rating/text consistency on training/validation data, then evaluate a restaurant-target-conditioned classifier with agreement and target-specific validation metrics; do not relabel this test case.

#### 9. yelp:test:10845 — hyperbole with possible label/text mismatch

Official label 1; predicted 0; P(positive) 0.000114; 13 processed tokens before the cap.

Evidence: “really pissed”; “fresh baked pecan chocolate chip cookies”; “someone needs to be shot for this.”.

**AI draft explanation:** The text literally complains that a favored item is unavailable, using an exaggerated threat. Fondness for the cookies could imply praise, but the overall positive official label is not explicit in the sentence. Treat sarcasm or rating-context mismatch as uncertain hypotheses rather than asserting either.

**One testable future fix:** Have independent annotators label hyperbolic training/validation reviews for both literal and intended polarity, then test a train-only augmentation that preserves those distinctions and report validation agreement and macro-F1.

#### 10. yelp:test:26683 — concessive polarity and negated low rating

Official label 1; predicted 0; P(positive) 0.000116; 17 processed tokens before the cap.

Evidence: “despite still not digging their ordering process”; “their food is just too good”; “disrespect with a 2 star review”.

**AI draft explanation:** The reviewer concedes an ordering complaint but explicitly rejects a low rating because the food is good. Negative words and the two-star reference occur inside that positive conclusion. Negation is retained by preprocessing, so the error concerns composition rather than removal of negation.

**One testable future fix:** Add training-only contrastive sentence pairs with despite/too-good-to/negated-rating constructions and evaluate on a prespecified compositional validation slice plus overall macro-F1.

### Near-threshold errors

#### 11. yelp:test:27612 — mixed aspect sentiment near threshold

Official label 0; predicted 1; P(positive) 0.500191; 44 processed tokens before the cap.

Evidence: “Terrible experience this time!”; “Very disappointed.”; “'slice of bacon from heaven' was truly from heaven”.

**AI draft explanation:** The main meal and service are criticized, followed by an emphatic exception praising bacon. Positive probability 0.50019 is essentially at the fixed decision boundary, consistent with conflicting aspects. One case cannot establish that changing the threshold would improve generalization.

**One testable future fix:** Compare an aspect-aware sentence aggregation head on independently annotated training/validation mixed reviews; select it using validation macro-F1 and report calibration without adjusting the threshold from this test packet.

#### 12. yelp:test:28852 — short idiomatic praise near threshold

Official label 1; predicted 0; P(positive) 0.499729; 5 processed tokens before the cap.

Evidence: “Try it it's like mom used to make”.

**AI draft explanation:** The very short recommendation expresses praise through a home-cooking idiom rather than an adjective such as great. Five processed tokens provide little context and the probability 0.49973 is marginally negative; the exact learned association is unknown.

**One testable future fix:** Add independently sourced training paraphrases of short idiomatic recommendations, then evaluate a frozen short-review validation slice and overall macro-F1 against the unchanged baseline.

#### 13. yelp:test:28068 — colloquial mixed sentiment and filtered emphasis

Official label 0; predicted 1; P(positive) 0.500644; 39 processed tokens before the cap.

Evidence: “The inside is amazing!!!!”; “@#$%& them for their cover”; “For a reason.”.

**AI draft explanation:** Interior/drink praise is mixed with criticism of the cover charge, payment facilities and clientele. Symbolic profanity and repeated punctuation are removed by the word tokenizer, potentially weakening expression; this is a plausible preprocessing limitation, not evidence of a causal ablation. The prediction is only slightly positive.

**One testable future fix:** Compare the original word tokenizer with train-fitted punctuation/emphasis indicator features on validation data, reporting mixed-review macro-F1 and ECE with a frozen threshold.

#### 14. yelp:test:23782 — temporal reversal across aspects near threshold

Official label 1; predicted 0; P(positive) 0.498702; 121 processed tokens before the cap.

Evidence: “the last two have been exceptional.”; “bland and unremarkable.”; “I'd happily recommend Lobby's”.

**AI draft explanation:** The review narrates a disappointing sandwich before two excellent burger visits and a recommendation. All 121 processed tokens fit the cap, so this is mixed temporal/aspect weighting rather than truncation. Probability 0.49870 indicates uncertain aggregation of the conflicting experiences.

**One testable future fix:** Train a sentence aggregation challenger with visit-order markers using training data and compare macro-F1 on a separate validation subset containing improvement across visits.

#### 15. yelp:test:19527 — restrained descriptive praise near threshold

Official label 1; predicted 0; P(positive) 0.498508; 17 processed tokens before the cap.

Evidence: “Simple, well prepared food.”; “Good atmosphere, good service.”; “Heavier and more traditional”.

**AI draft explanation:** The short review clearly praises preparation, atmosphere and service; heavier/traditional is a descriptive comparison rather than an explicit complaint. A 0.49851 probability is consistent with uncertainty over these neutral descriptors, but token contribution cannot be inferred from the score alone.

**One testable future fix:** Collect train/validation examples of restrained praise with neutral comparative descriptors and test balanced training-only augmentation, measuring short-review macro-F1 and calibration on validation.

### Long-review slice errors

#### 16. yelp:test:10081 — long mixed review with advisory negative language

Official label 1; predicted 0; P(positive) 0.039683; 581 processed tokens before the cap.

Evidence: “I don't have a single complaint”; “You can still get a quality room”; “everybody is happy”.

**AI draft explanation:** The overall recommendation is positive, but instructions include warnings about poor rooms, queues and expectations. At 581 processed tokens, the review exceeds the 384-token prefix. Literal warning vocabulary and missing later positive context are plausible contributors; neither is confirmed without an ablation.

**One testable future fix:** Compare a fixed head-and-tail encoding with the current prefix on training/validation data, and report validation long-review macro-F1 plus retained positive-advisory examples under the same threshold.

#### 17. yelp:test:4664 — long temporal deterioration with context loss

Official label 0; predicted 1; P(positive) 0.930167; 570 processed tokens before the cap.

Evidence: “Great food, cheap beer, attentive service”; “Zero stars.”; “you'll want to go somewhere else.”.

**AI draft explanation:** A glowing first visit is followed by extensive poor service and a negative final recommendation. The 570 processed tokens exceed the input cap, so the final verdict can be lost while early praise remains. The model's positive output fits this concern, but long-form aggregation also remains a candidate explanation.

**One testable future fix:** Evaluate first-384 versus fixed head-and-tail representations in a train/validation ablation, with a prespecified long temporal-deterioration validation slice and overall macro-F1.

#### 18. yelp:test:9911 — long mixed review with positive conclusion beyond prefix

Official label 1; predicted 0; P(positive) 0.002201; 563 processed tokens before the cap.

Evidence: “The tables here are cursed”; “These people really made our trip memorable”; “I'm giving this place 4 stars”.

**AI draft explanation:** The review includes criticism of gambling, drinks and food, then praises housekeeping, room service and accommodating customer service with an explicit four-star conclusion. Its 563 processed tokens exceed the cap; later context may be absent from the model input. Attribution to truncation remains a testable hypothesis.

**One testable future fix:** Compare head-and-tail token allocation with the current prefix using identical training/validation splits, reporting long-review macro-F1, calibration and timing before any new held-out evaluation.

#### 19. yelp:test:9919 — long mixed review with late service-based reversal

Official label 1; predicted 0; P(positive) 0.037934; 536 processed tokens before the cap.

Evidence: “Do I have complaints?  PLENTY.”; “great treatment of dumbass me merited another star”; “THAT'S SERVICE.”.

**AI draft explanation:** A long catalogue of hotel shortcomings is offset by praise for inexpensive accommodations, entertainment and especially assistance after an injury. The 536-token review exceeds the cap, making the detailed late service narrative vulnerable to truncation. The first sentence already signals the improved rating, so truncation is not a complete explanation by itself.

**One testable future fix:** Run a train/validation comparison of hierarchical sentence pooling with prefix pooling, evaluating long reviews with final positive service events and reporting overall and slice macro-F1.

#### 20. yelp:test:26052 — sarcastic long-form setup and positive late verdict

Official label 1; predicted 0; P(positive) 0.035674; 461 processed tokens before the cap.

Evidence: “we need another burger joint, like earth needs global warming”; “Zinburger's best, by far”; “i'm giving it to Zin”.

**AI draft explanation:** A long ironic poem criticizes the proliferation of burger restaurants before recommending this one. The 461 processed tokens exceed the cap and OOV rate is about 6.5%, so unusual wording and context loss may compound the irony. Sarcasm and the role of OOV are hypotheses, not measured causal findings.

**One testable future fix:** Compare a train-only subword tokenizer plus fixed head-and-tail representation with the word-prefix baseline using an independently annotated validation sarcasm/long-review slice, reporting macro-F1 and vocabulary/resource cost.

## dilated_cnn

Selected best checkpoint: `f3883a005a369ef495c99bebe3cd51c4a7858c3f58121b6dbcfecee341dd8ac2`.

### Confident false positives

#### 1. yelp:test:9938 — Mixed aspects: value and location versus hygiene

Official label 0; predicted 1; P(positive) 0.999979; 113 processed tokens before the cap.

Evidence: “you can't beat it for the price and location”; “the drains in the shower and the sink backed up”.

**AI draft explanation:** The review praises affordability and location but describes backed-up drains and dirty water. All 113 tokens fit in the input. Positive value phrases may outweigh the hygiene complaint in pooled CNN features; this is a hypothesis, not a demonstrated causal attribution.

**One testable future fix:** Compare global pooling with sentence-level aspect attention using the same training split and seed. Prespecify a validation slice contrasting price/location praise with hygiene complaints; select using its macro-F1 and overall validation macro-F1.

#### 2. yelp:test:5752 — Updated negative judgment followed by historical praise

Official label 0; predicted 1; P(positive) 0.999972; 260 processed tokens before the cap.

Evidence: “This was a 4-star review”; “customer service have gone down the tubes”.

**AI draft explanation:** An opening update retracts an earlier four-star judgment, while the older body praises the restaurant. All 260 tokens are retained. The CNN may give more weight to extensive historical praise than the current update, but an attribution or ablation would be needed to establish that mechanism.

**One testable future fix:** Train a position-aware sentence-attention variant on training reviews, preserving update markers. Compare it with the original CNN on a manually audited validation update/temporal-contrast slice and overall validation macro-F1 before selecting a variant.

#### 3. yelp:test:29330 — Apparent text-label ambiguity in an overtly positive short review

Official label 0; predicted 1; P(positive) 0.999968; 23 processed tokens before the cap.

Evidence: “Wow love the place”; “Great place to come and relax”.

**AI draft explanation:** The complete 23-token review is favorable and gives no clear complaint. Its supplied negative label is difficult to explain from visible text alone. This may reflect missing rating context or label ambiguity; it does not prove an annotation error, and the fixed test label must stay unchanged.

**One testable future fix:** Blindly audit a prespecified sample of similar short training/validation reviews for text-rating disagreement. Compare a documented training-only label-quality rule or noise-robust loss with the original loss, selecting by validation macro-F1 and calibration; do not relabel this test example.

#### 4. yelp:test:23815 — Uneven aspect sentiment and possible text-rating ambiguity

Official label 0; predicted 1; P(positive) 0.999880; 129 processed tokens before the cap.

Evidence: “The bakery here is tops”; “The deli isn't the greatest”.

**AI draft explanation:** Bakery praise dominates a review that qualifies the deli and grocery selection. All 129 tokens fit. Global pooled features may overemphasize the bakery, although the mostly favorable text also makes the negative label ambiguous; neither interpretation is proven.

**One testable future fix:** Train a sentence-level attention variant using the same training vocabulary and seed. Compare it with the original CNN on a manually audited validation slice of reviews covering products, selection and service; choose using validation macro-F1 and preserve test labels.

#### 5. yelp:test:17407 — Visit reversal with a truncated negative conclusion

Official label 0; predicted 1; P(positive) 0.999878; 453 processed tokens before the cap.

Evidence: “it will probably be the last”; “Bad place for someone”.

**AI draft explanation:** The 453-token review contrasts a favorable first visit with a disappointing changed restaurant. The phrase beginning Bad place for someone starts at tokenizer index 439, beyond the 384-token prefix. Earlier negative evidence remains, so the omitted conclusion is a plausible contributor rather than proof of the cause.

**One testable future fix:** Retrain a controlled head-288/tail-96 input variant at the same 384-token budget, seed and schedule. Compare long-review validation macro-F1/error rate and overall validation macro-F1 against prefix-only encoding; do not choose the input rule using this test case.

### Confident false negatives

#### 6. yelp:test:22807 — Vague update followed by retained negative service narrative

Official label 1; predicted 0; P(positive) 0.000012; 60 processed tokens before the cap.

Evidence: “They really did change the service up”; “Horrible service.”.

**AI draft explanation:** A short edit signals a change, while the rest describes a very bad earlier experience. All 60 tokens are retained. The positive label may depend on the edit or rating context that is not explicit in the text; the negative prediction may follow the detailed complaint. This does not establish which interpretation is correct.

**One testable future fix:** Manually audit analogous training/validation updates without using the fixed test labels to tune decisions. Train an update-aware sentence encoder and compare its validation update-slice macro-F1 with the original CNN, keeping all reported test labels and predictions unchanged.

#### 7. yelp:test:30793 — Past-versus-current owner contrast

Official label 1; predicted 0; P(positive) 0.000301; 64 processed tokens before the cap.

Evidence: “This place is so much better since they changed owners.”; “It was horrible.”; “Now its much better.”.

**AI draft explanation:** The review explicitly separates a horrible former owner from better current ownership. All 64 tokens fit, so truncation cannot explain this case. The CNN may fail to bind negative phrases to the past and positive phrases to the present; this is a mechanism hypothesis.

**One testable future fix:** Train a position-aware pooling variant and compare it with the original CNN on a prespecified validation ownership-change slice. Use training examples for temporal-contrast augmentation; select the variant by validation results and use removal of past-episode text only as a diagnostic, not a new headline test score.

#### 8. yelp:test:26683 — Concessive praise with negated low-rating language

Official label 1; predicted 0; P(positive) 0.000408; 17 processed tokens before the cap.

Evidence: “their food is just too good to disrespect with a 2 star review”.

**AI draft explanation:** The 17-token review praises the food while mentioning an undeserved two-star review. A negative rating phrase may be represented without the construction that rejects it, but no causal attribution has been performed. The full review is present in the input.

**One testable future fix:** Add label-checked concessive and rating-negation examples to training only, holding the architecture and schedule fixed. Compare validation macro-F1 on a prespecified concessive/rating-negation slice and overall validation macro-F1 before choosing augmentation.

#### 9. yelp:test:21215 — Sentiment toward other patrons versus the business response

Official label 1; predicted 0; P(positive) 0.000730; 100 processed tokens before the cap.

Evidence: “the bartender was very cordial”; “had mind to do the right thing”.

**AI draft explanation:** The writer criticizes parents bringing children into a bar while approving the bartender response. All 100 tokens fit. Strong negative language about third parties may be conflated with sentiment about the business; this target-confusion explanation remains unverified.

**One testable future fix:** Annotate sentiment targets in a training/validation subset and compare a target-aware attention variant with global pooling. Prespecify a validation slice where patrons receive criticism but staff receive praise, and select using its macro-F1 plus overall validation macro-F1.

#### 10. yelp:test:10845 — Complaint about a valued product with ambiguous overall polarity

Official label 1; predicted 0; P(positive) 0.001060; 13 processed tokens before the cap.

Evidence: “they don't have the fresh baked pecan chocolate chip cookies anymore”.

**AI draft explanation:** The complete 13-token review complains that an apparently appreciated product is unavailable. Its positive supplied label is not clearly explained by the visible complaint. Product appreciation and overall satisfaction may be conflated, or rating context may be missing; neither possibility is established.

**One testable future fix:** Audit similar short product-availability reviews in training/validation, then compare target-aware phrase features with the original CNN. Select using validation macro-F1 and calibration, document ambiguity, and retain the fixed test label rather than treating it as a correction target.

### Near-threshold errors

#### 11. yelp:test:11322 — Borderline mixed recommendation and rating qualification

Official label 1; predicted 0; P(positive) 0.499721; 116 processed tokens before the cap.

Evidence: “The food is consistently good”; “can't seem to give them five stars”.

**AI draft explanation:** The review mixes food/service praise with qualifications about the rating and returning. The positive probability is about 0.49972, very close to the fixed 0.5 threshold. This is a borderline decision; the text alone does not establish a single causal feature.

**One testable future fix:** Choose one global decision threshold using validation macro-F1 only and report its validation calibration trade-off. Preserve the original fixed-threshold test results; if a validation-selected threshold is later evaluated, label it as a separate prespecified experiment rather than tuning to this test case.

#### 12. yelp:test:2116 — Counterfactual courtesy implies an estimator did not arrive

Official label 0; predicted 1; P(positive) 0.500325; 22 processed tokens before the cap.

Evidence: “would've shown up”; “wait around thinking”.

**AI draft explanation:** The 22-token review describes appreciation that would have been possible if an estimator had arrived or called. The negative event is implicit in a counterfactual construction, while surface courtesy words may suggest approval. The full input is retained, and feature attribution is still needed to establish the cause.

**One testable future fix:** Add label-checked no-show and counterfactual-courtesy examples to training only. Compare the unchanged CNN against this augmentation on a prespecified validation counterfactual slice and overall macro-F1 before selecting the training recipe.

#### 13. yelp:test:17408 — Amenity praise inside an overall negative mall experience

Official label 0; predicted 1; P(positive) 0.500393; 192 processed tokens before the cap.

Evidence: “WHO BUILDS AN OUTDOOR MALL IN THE MIDDLE OF THE FREAKIN' DESERT!”; “the best part of my trip to this mall was leaving”.

**AI draft explanation:** The review criticizes heat, navigation and broken misters while mentioning a nice play area. All 192 tokens fit, so this is not an input-cutoff case. Local positive amenities may outweigh the negative overall assessment in pooled features; this remains a hypothesis.

**One testable future fix:** Compare sentence-level attention with global pooling using the same training seed and schedule. Prespecify a validation mixed-amenity slice and measure its macro-F1/error rate plus overall validation macro-F1; choose the pooling rule without revisiting fixed test predictions.

#### 14. yelp:test:19294 — Sarcastic conditional praise and quoted fortune-cookie language

Official label 0; predicted 1; P(positive) 0.502173; 49 processed tokens before the cap.

Evidence: “If you like bad chinese food, you'll love this place.”; “Have fun tonight, indeed.”.

**AI draft explanation:** The review uses an ironic conditional and echoes a fortune-cookie message after a stomach complaint. All 49 tokens fit. Positive words are embedded in sarcasm and quotation; the CNN may lose that pragmatic context, but the prediction does not prove a particular feature caused it.

**One testable future fix:** Compare a training/validation preprocessing variant that preserves quotation and contrast boundaries with the frozen tokenizer. Train both from scratch at the same seed/schedule, and select using a prespecified validation sarcasm/conditional slice plus overall macro-F1; keep the original test result intact.

#### 15. yelp:test:13231 — Positive facilities versus negative patron and staff experience

Official label 0; predicted 1; P(positive) 0.502511; 124 processed tokens before the cap.

Evidence: “There are plenty of downsides, though.”; “There are plenty of better megaplexes around here.”.

**AI draft explanation:** Stadium seating and prices receive praise, but rude patrons, staff inaction and a recommendation to use alternatives drive the negative judgment. All 124 tokens fit. Global feature pooling may underweight the final comparative recommendation, but that explanation needs controlled testing.

**One testable future fix:** Train a sentence-level attention variant and compare it with the original CNN on validation reviews that contrast facilities with staff/service complaints. Keep the vocabulary, seed and schedule fixed, and select using slice and overall validation macro-F1.

### Long-review slice errors

#### 16. yelp:test:4664 — Visit reversal with severe complaints beyond the input cutoff

Official label 0; predicted 1; P(positive) 0.994477; 570 processed tokens before the cap.

Evidence: “two completely different restaurants”; “Zero stars.”; “go somewhere else.”.

**AI draft explanation:** The 570-token review contrasts a good first visit with a very bad second visit. Zero stars begins at tokenizer index 504 and go somewhere else at 558, outside the 384-token prefix. Some earlier complaints are retained, so truncation is a plausible contributor rather than a proven explanation of the confident false positive.

**One testable future fix:** Compare head-288/tail-96 encoding with prefix-only encoding at the same 384-token budget, seed and training schedule. Select using long-review validation macro-F1/error rate and overall macro-F1; report any throughput change.

#### 17. yelp:test:9911 — Positive service recovery omitted from a long review

Official label 1; predicted 0; P(positive) 0.004710; 563 processed tokens before the cap.

Evidence: “brand new penthouse”; “These people really made our trip memorable”.

**AI draft explanation:** The 563-token review includes complaints before a manager resolves the problem with an upgraded room. The penthouse phrase begins at tokenizer index 521 and the memorable-trip conclusion at 529, beyond the prefix limit. Lost recovery context is plausible, but truncation alone has not been causally demonstrated.

**One testable future fix:** Train a paragraph-chunk pooling model from scratch on training data and compare it with the original fixed-prefix CNN. Select using long-review and recovery-slice validation macro-F1 while recording runtime and peak memory; use no test cases to choose chunking parameters.

#### 18. yelp:test:37318 — Anticipated praise versus actual disappointment in a long review

Official label 0; predicted 1; P(positive) 0.788689; 561 processed tokens before the cap.

Evidence: “i was super excited”; “YU SHOuld not go to this place.”; “The only thing elevated about this place is the price.”.

**AI draft explanation:** The 561-token review opens with enthusiasm for an imagined meal, then criticizes actual food and management while praising some service. The 384-token prefix omits part of the later review. Temporal contrast and omitted context are plausible contributors, but the exact retained evidence and model attribution would be needed to establish causality.

**One testable future fix:** Evaluate head/tail encoding and position-aware sentence pooling in separate controlled training experiments with the same seed/schedule. Prespecify a validation anticipated-versus-actual-experience slice and long-review macro-F1; choose each change using validation only.

#### 19. yelp:test:29374 — Food praise followed by lengthy service and management complaints

Official label 0; predicted 1; P(positive) 0.613981; 547 processed tokens before the cap.

Evidence: “Even the best food and atmosphere can taste horrendous”; “There ARE better AYCE in town”.

**AI draft explanation:** The 547-token review praises food while describing delays, a frozen dish and poor management. The closing alternative recommendation starts at tokenizer index 536 and is omitted by the prefix limit. Negative evidence also occurs earlier, so a lost conclusion is not a complete proven explanation.

**One testable future fix:** Retrain with head-256/tail-128 encoding at the same 384-token budget and compare with prefix-only input, keeping the seed and schedule fixed. Select on mixed-aspect and long-review validation macro-F1, overall macro-F1 and throughput.

#### 20. yelp:test:9919 — Positive staff recovery after hotel complaints in a long review

Official label 1; predicted 0; P(positive) 0.030189; 536 processed tokens before the cap.

Evidence: “their great treatment of dumbass me merited another star”; “they were all VERY nice to me.”; “THAT'S SERVICE.”.

**AI draft explanation:** The 536-token review contains hotel complaints but awards an additional star for staff help after an injury. Detailed kindness and the service conclusion start at tokenizer indices 467 and 535, beyond the prefix limit. An opening recovery summary remains visible, so omitted detail may contribute without fully explaining the false negative.

**One testable future fix:** Compare head-256/tail-128 encoding with the original prefix-only CNN at the same seed, schedule and token budget. Prespecify a validation service-recovery slice and long-review macro-F1; choose using validation results and record examples/second alongside accuracy.

## Student review and report work still pending

Open each full source packet, compare the label/prediction with the complete
review and retained model input, then accept or revise the AI hypothesis in
your own words. Record an evidence-based error type, explanation and testable
fix; mark `student_reviewed=True` only after that review actually occurs.
The team still needs independent teammate results, cross-member comparison,
viva understanding and one combined three-part report/submission. Assisted
analysis is disclosed in root `AI_USE.md` and is not proof of independent
student authorship or completion of the manual requirement.
