# Part 2 - AI-assisted error-analysis draft

This document renders 60 already-authored AI-assisted hypotheses: twenty actual test errors for each required model. Exact case IDs, quotes, text hashes, packet hashes and checkpoint identities were checked before rendering. It does not certify independent student analysis or a causal explanation.

Each packet contains five confident false positives, five confident false negatives, five near-threshold errors and five prespecified long-review slice errors. IDs are distinct within a model; texts can overlap across models. These selected cases cannot estimate the prevalence of error types.

Current evaluation publication: `reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/selected`. Frozen selection SHA-256: `4eedab69b63b4c064283ab44c6ca6f8c4d19ea14d81c0c5b3941a599abbbc8b9`.

Selection rule recorded by the completed evaluation:

```text
For each family, consider its existing desktop control and every completed new candidate. Find the highest selected-checkpoint validation macro-F1. Among candidates within 0.001 inclusive of that maximum, prefer fewer parameters, then higher validation macro-F1, then source path and candidate ID. A wider candidate must therefore exceed a smaller control by more than 0.001 to displace it. Checkpoints follow configured early_stopping_min_delta=0.0001. No test score or test prediction enters selection. This practical tolerance does not establish statistical significance.
```

Historical official-test scores and texts have already been observed; this repeated evaluation is not a newly sealed holdout. Checkpoint selection uses validation. The proposals below are future training/validation studies with appropriate new evaluation, not permission to tune on these test errors or alter test labels.

Observed packet metadata: **0 / 60 `student_reviewed` flags are set**, and 0 flagged rows also have both required human interpretation fields filled. The renderer preserves those fields and the notebook byte-for-byte. Recorded flags are not independent verification that a human performed the review.

## Current measured comparison

| Model | Completed / selected epoch | Validation macro-F1 | Test accuracy | Test macro-F1 | Brier | ECE (15 bins) | Errors / test rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| maxpool_mlp | 9 / 5 | 0.928285 | 93.3105% | 0.933105 | 0.049852 | 0.013869 | 2,542 / 38,000 |
| bilstm | 12 / 11 | 0.958857 | 96.1211% | 0.961210 | 0.029803 | 0.014531 | 1,474 / 38,000 |
| dilated_cnn | 6 / 6 | 0.953446 | 95.6263% | 0.956262 | 0.033397 | 0.012759 | 1,662 / 38,000 |

These are the actual selected checkpoints' measurements. The table does not establish a global optimum, multi-seed ranking or production readiness. Bootstrap intervals in the complete metric files concern test-row sampling rather than training-seed uncertainty.

| Experimental model vs MLP | Accuracy difference (percentage points) | Exact paired McNemar p-value |
|---|---:|---:|
| bilstm | +2.8105 | 3.62256e-118 |
| dilated_cnn | +2.3158 | 2.80945e-85 |

The two stored baseline comparisons are unadjusted paired tests on the same examples; they do not compare the two experimental models directly or prove an architectural cause.

The duplicate audit records 7 train/test shared raw-text hashes. Preserve and disclose the official-row protocol rather than treating it as a duplicate-clean benchmark.

| Model | Prefix token cap | Recorded training CPU | Recorded training GPU |
|---|---:|---|---|
| maxpool_mlp | 384 | Intel(R) Core(TM) Ultra 9 285K | NVIDIA GeForce RTX 5090 |
| bilstm | 384 | Intel(R) Core(TM) Ultra 9 285K | NVIDIA GeForce RTX 5090 |
| dilated_cnn | 384 | Intel(R) Core(TM) Ultra 9 285K | NVIDIA GeForce RTX 5090 |

Predicted probabilities and review text alone do not identify causal mechanisms. Mixed sentiment, target attribution, temporal changes, irony, truncation and label/text mismatch are hypotheses to investigate. The prefix cap can omit context, but a long review alone does not establish why a prediction failed.

## maxpool_mlp

Training source: `reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search/training/mlp_wide_regularized`. Selected checkpoint SHA-256: `5ae7df4f747e89209ec2ca1381652bf3ed0fcf671a6f15605cb08323a0d9599c`.

Recorded reviewed flags: 0 / 20. Every explanation below remains labeled as an AI-assisted draft.

### Confident false positives

#### 1. yelp:test:29330 - Possible text-label disagreement

Official label 0; prediction 1; P(positive) 0.999998; 23 processed tokens before truncation.

Exact evidence:

```text
Wow love the place
```

```text
Great place to come and relax worth a try!
```

**AI-assisted hypothesis:** The short review praises the venue, cleanliness and atmosphere without stating a complaint. Its negative label is therefore difficult to explain from the visible text. This could reflect missing context or label noise, but the fixed label should not be changed.

**Proposed future training/validation study:** Test a training-only label-quality filter using an independently reviewed validation sample.

#### 2. yelp:test:20681 - Ambiguous short review and overconfidence

Official label 0; prediction 1; P(positive) 0.999988; 2 processed tokens before truncation.

Exact evidence:

```text
Large variety
```

**AI-assisted hypothesis:** The entire review contains only two words. They could express approval or simply describe the selection, with no clear complaint to explain the negative label. The almost certain positive prediction appears too confident for such limited context.

**Proposed future training/validation study:** Test a short-review abstention threshold using validation coverage and error rate.

#### 3. yelp:test:25701 - Mixed sentiment and missed contrast

Official label 0; prediction 1; P(positive) 0.999983; 46 processed tokens before truncation.

Exact evidence:

```text
good.....but not great
```

```text
there are much better options
```

**AI-assisted hypothesis:** The review praises the drinks, cleanliness and service but says the main food is only adequate and better options exist. Max pooling may favor the many positive words without preserving the contrast that supports a less favorable overall judgment.

**Proposed future training/validation study:** Compare a bigram-aware MLP with the current baseline on validation reviews containing contrast phrases.

#### 4. yelp:test:12480 - Value complaint within mostly positive wording

Official label 0; prediction 1; P(positive) 0.999953; 13 processed tokens before truncation.

Exact evidence:

```text
a little on the small side for $10
```

```text
Our server was awesome!
```

**AI-assisted hypothesis:** Most wording praises the food and server, while the main criticism concerns a small sandwich for its price. The model may emphasize those positive words. The negative label could reflect the value complaint or context not fully stated in this short review.

**Proposed future training/validation study:** Compare mean-plus-max pooling against max pooling on mixed-sentiment validation reviews.

#### 5. yelp:test:1560 - Sarcasm and removed emoticon

Official label 0; prediction 1; P(positive) 0.999950; 14 processed tokens before truncation.

Exact evidence:

```text
awesome if you want to wear your food
```

```text
better than eating it. :(
```

**AI-assisted hypothesis:** The apparently positive words are sarcastic: the burrito falls apart, and wearing it is described as preferable to eating it. The tokenizer removes the sad emoticon. Max pooling may preserve words such as 'awesome' while missing the sarcastic relationship between clauses.

**Proposed future training/validation study:** In a future run, retain emoticon tokens and measure validation F1 on sarcastic reviews.

### Confident false negatives

#### 6. yelp:test:1131 - Informal negation and idiom

Official label 1; prediction 0; P(positive) 0.000001; 4 processed tokens before truncation.

Exact evidence:

```text
wont
```

```text
do ya dirty
```

**AI-assisted hypothesis:** The phrase says the drinks will not disappoint, but uses informal spelling. The tokenizer does not expand 'wont' into a negation, and the isolated word 'dirty' may push this model toward negative sentiment without the meaning of the whole phrase.

**Proposed future training/validation study:** Add 'wont' to contraction handling in a future run and check the validation negation slice.

#### 7. yelp:test:30958 - Complaint followed by a positive resolution

Official label 1; prediction 0; P(positive) 0.000003; 235 processed tokens before truncation.

Exact evidence:

```text
why do I give this store 5 stars?
```

```text
my opinion of Cox has changed.
```

**AI-assisted hypothesis:** The writer describes a long dispute with the company before praising a store manager who resolved it. The positive rating concerns that resolution. Max pooling may retain strong complaint words without distinguishing the earlier problem from the later change in opinion.

**Proposed future training/validation study:** Test paragraph pooling with extra weight on the final paragraph using validation data.

#### 8. yelp:test:22807 - Review edit mixed with an older complaint

Official label 1; prediction 0; P(positive) 0.000192; 60 processed tokens before truncation.

Exact evidence:

```text
EDIT: They really did change the service up since I last posted this.
```

```text
Horrible service.
```

**AI-assisted hypothesis:** The opening edit suggests a change in service, while the retained older review is strongly negative. Max pooling may be dominated by that older wording. The brief update may explain the positive label, but the text alone does not confirm the rating history.

**Proposed future training/validation study:** Test an update-aware representation on validation reviews with clearly marked edits.

#### 9. yelp:test:30793 - Past and present sentiment confused

Official label 1; prediction 0; P(positive) 0.000263; 64 processed tokens before truncation.

Exact evidence:

```text
It was horrible.
```

```text
Now its much better.
```

**AI-assisted hypothesis:** The review contrasts a bad visit under former owners with friendly service under new owners. The current verdict is positive. Max pooling may keep strong words from the old experience without preserving which time period they describe.

**Proposed future training/validation study:** Test sentence-position features on before-and-after reviews in validation.

#### 10. yelp:test:30354 - Qualified praise with comparative criticism

Official label 1; prediction 0; P(positive) 0.000405; 66 processed tokens before truncation.

Exact evidence:

```text
they are just good in my opinion
```

```text
this is a local 4
```

**AI-assisted hypothesis:** The writer considers the restaurant good locally but less impressive than its reputation or restaurants elsewhere. Several critical words appear alongside an explicit local four-star judgment. The model may give those criticisms more weight than the qualified positive conclusion.

**Proposed future training/validation study:** Compare mean-plus-max pooling on validation reviews that mix praise with mild criticism.

### Near-threshold errors

#### 11. yelp:test:22922 - Other restaurants confused with the reviewed venue

Official label 1; prediction 0; P(positive) 0.499898; 186 processed tokens before truncation.

Exact evidence:

```text
No more gross, greasy, no flavor Mexican food for me!
```

```text
It was delicious!
```

**AI-assisted hypothesis:** Much of the opening criticizes other restaurants and regional food styles. The actual visit is described positively, with praise for both food and service. The score falls just below the threshold, possibly because the model mixes those different targets.

**Proposed future training/validation study:** Evaluate sentence-level attention on validation reviews comparing several restaurants.

#### 12. yelp:test:20477 - Negated complaint and secondary criticism

Official label 1; prediction 0; P(positive) 0.499824; 66 processed tokens before truncation.

Exact evidence:

```text
I don't think this one is overpriced.
```

```text
ridiculous parking system.
```

**AI-assisted hypothesis:** Most of the review praises the food and healthy options. It also rejects an overpricing complaint and criticizes the parking system rather than the restaurant. The score sits just below the threshold; the model may mix these separate points instead of preserving the overall approval.

**Proposed future training/validation study:** Test bigram features on validation examples with negated complaints.

#### 13. yelp:test:35815 - Improvement and resolved problems

Official label 1; prediction 0; P(positive) 0.499429; 34 processed tokens before truncation.

Exact evidence:

```text
I see improvement in all directions.
```

```text
look anymore..
```

**AI-assisted hypothesis:** The review describes friendlier staff, good food and an improved experience. It mentions an earlier problem only to say it has gone away. The near-threshold negative prediction may reflect difficulty separating resolved criticism from the current positive assessment.

**Proposed future training/validation study:** Compare a sequence encoder on validation reviews describing improvements and resolved problems.

#### 14. yelp:test:7121 - Indirect complaint mixed with minor positives

Official label 0; prediction 1; P(positive) 0.500846; 38 processed tokens before truncation.

Exact evidence:

```text
Its confusing
```

```text
I wish the Airport was a little more user-friendly.
```

**AI-assisted hypothesis:** The main complaints are confusing pickup arrangements and a long walk. Positive language about moving walkways and flying instead of driving is secondary. The score barely exceeds the threshold, suggesting the overall criticism was not clearly separated from those positives.

**Proposed future training/validation study:** Test phrase features on validation reviews containing indirect complaints such as 'wish' statements.

#### 15. yelp:test:37435 - Failed service despite positive planning language

Official label 0; prediction 1; P(positive) 0.500865; 16 processed tokens before truncation.

Exact evidence:

```text
Was not able to eat here for lunch
```

```text
despite perfect planning.
```

**AI-assisted hypothesis:** This is a complaint about being unable to get seating on two days. 'Perfect planning' describes the customer's preparation, not the restaurant. The near-threshold positive score may reflect that approving phrase without preserving the repeated service failure.

**Proposed future training/validation study:** Compare a phrase-aware encoder on validation reviews containing 'not able' service complaints.

### Long-review slice errors

#### 16. yelp:test:10081 - Advice mixed with criticism and truncated praise

Official label 1; prediction 0; P(positive) 0.391200; 581 processed tokens before truncation.

Exact evidence:

```text
I don't have a single complaint
```

```text
The Flamingo pool scene is good times
```

**AI-assisted hypothesis:** The review recommends the hotel while warning about cheaper rooms and check-in problems. Negative words in that advice may dominate max pooling. The 384-token cutoff also removes later praise of the pool, although the positive opening is still visible.

**Proposed future training/validation study:** Compare the first 192 plus last 192 tokens against the first 384 tokens on long validation reviews.

#### 17. yelp:test:9911 - Positive conclusion lost after truncation

Official label 1; prediction 0; P(positive) 0.178834; 563 processed tokens before truncation.

Exact evidence:

```text
The staff handled the situation pretty well actually.
```

```text
I'm giving this place 4 stars
```

**AI-assisted hypothesis:** The visible portion mixes praise with gambling and food complaints. The final four-star verdict and several later examples of helpful service fall beyond the 384-token limit. Missing that ending is a plausible contributor, although earlier positive evidence is still available.

**Proposed future training/validation study:** Test pooling across all review chunks and compare validation F1 for long reviews.

#### 18. yelp:test:37318 - Imagined praise and truncated negative conclusion

Official label 0; prediction 1; P(positive) 0.775001; 561 processed tokens before truncation.

Exact evidence:

```text
YU SHOuld not go to this place.
```

```text
Overpriced, non elevated bad  food
```

**AI-assisted hypothesis:** The opening spends many words on an imagined enjoyable meal before describing the disappointing visit. Some later complaints and the final negative summary are truncated. Max pooling may favor the early enthusiastic words, despite clear negative statements that remain in the input.

**Proposed future training/validation study:** Test paragraph pooling across the full review, selected on long validation reviews.

#### 19. yelp:test:9919 - Positive resolution lost after truncation

Official label 1; prediction 0; P(positive) 0.204882; 536 processed tokens before truncation.

Exact evidence:

```text
merited another star.
```

```text
THAT'S SERVICE.
```

**AI-assisted hypothesis:** The retained text contains many hotel complaints, while the detailed account of helpful staff after an accident appears later. The 384-token limit drops that positive resolution. The opening still signals a higher rating, so truncation alone cannot explain the error.

**Proposed future training/validation study:** Compare tokens from both the beginning and end against the current cutoff on long validation reviews.

#### 20. yelp:test:26354 - Final positive judgment lost after truncation

Official label 1; prediction 0; P(positive) 0.220817; 524 processed tokens before truncation.

Exact evidence:

```text
The burger was good.
```

```text
That soup alone brought the 3-star lunch up to a 4-star lunch.
```

**AI-assisted hypothesis:** The review praises the setting and service but criticizes details of the burger. Its decisive soup praise and final four-star rating appear after the 384-token cutoff. Their omission may leave the model with a less positive balance than the complete review.

**Proposed future training/validation study:** Compare pooling across all review chunks with the current truncation on validation, keeping the training split fixed.

## bilstm

Training source: `reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/training/bilstm`. Selected checkpoint SHA-256: `0758cbe7405758fe6f99079f02fd3359aa1367d120cdd984c361c03a34fd54ba`.

Recorded reviewed flags: 0 / 20. Every explanation below remains labeled as an AI-assisted draft.

### Confident false positives

#### 1. yelp:test:29330 - Possible text-label mismatch

Official label 0; prediction 1; P(positive) 0.999996; 23 processed tokens before truncation.

Exact evidence:

```text
Wow love the place
```

```text
worth a try!
```

**AI-assisted hypothesis:** The review praises the place, cleanliness and atmosphere, while its official label is negative. The prediction disagrees with the label, but the visible text does not clearly explain why. The label is retained.

**Proposed future training/validation study:** Audit a fixed training/validation sample for rating-text disagreement with independent annotators; compare validation results before and after correcting confirmed training-label issues.

#### 2. yelp:test:17407 - Mixed visits and possible context loss

Official label 0; prediction 1; P(positive) 0.999929; 453 processed tokens before truncation.

Exact evidence:

```text
It was fantastic.
```

```text
SOOOOOO TINY.
```

```text
Bad place for someone who hasn't eaten
```

**AI-assisted hypothesis:** Praise for the first visit contrasts with disappointment about the new menu and small portions. The review exceeds the 384-token limit, so some later criticism may be omitted. Early praise could also outweigh the complaints.

**Proposed future training/validation study:** Compare the first 384 tokens with 192 opening plus 192 closing tokens, using the same training setup and long-review validation slice.

#### 3. yelp:test:9706 - Mixed aspects and qualified praise

Official label 0; prediction 1; P(positive) 0.999900; 164 processed tokens before truncation.

Exact evidence:

```text
very beautiful
```

```text
my feet are still hurting
```

```text
if someone else was paying the bill
```

**AI-assisted hypothesis:** The rooms and staff receive praise, but navigation, convenience and value receive criticism. The writer would return only if someone else paid. The whole processed review fits the input limit, so truncation is not the issue here.

**Proposed future training/validation study:** Compare mean pooling with the current max pooling, keeping the BiLSTM unchanged, and measure macro-F1 on mixed-review validation examples.

#### 4. yelp:test:4655 - Mixed sentiment and possible irony

Official label 0; prediction 1; P(positive) 0.999888; 34 processed tokens before truncation.

Exact evidence:

```text
good drink selection.
```

```text
bad: the parking and the crowd.
```

```text
totally the coolest place, like, ever.
```

**AI-assisted hypothesis:** The review lists both good and bad points and then mocks the crowd. Literal positive phrases may outweigh the complaints. The degree of irony is uncertain, so this example does not establish a general sarcasm weakness.

**Proposed future training/validation study:** Add labeled good/bad contrast examples from training data and compare performance on a separate mixed-sentiment validation slice.

#### 5. yelp:test:5752 - Updated opinion versus old praise

Official label 0; prediction 1; P(positive) 0.999834; 260 processed tokens before truncation.

Exact evidence:

```text
This was a 4-star review
```

```text
have gone down the tubes
```

```text
All wonderful.
```

**AI-assisted hypothesis:** The opening note says food and service have declined, while most of the body describes an earlier positive visit. The input is not truncated. The positive prediction may give too much weight to the older praise.

**Proposed future training/validation study:** Add training examples with explicit review updates and evaluate the retrained model on a fixed validation subset containing updates.

### Confident false negatives

#### 6. yelp:test:21215 - Wrong sentiment target

Official label 1; prediction 0; P(positive) 0.000030; 100 processed tokens before truncation.

Exact evidence:

```text
they completely took advantage of her
```

```text
the bartender was very cordial
```

```text
do the right thing
```

**AI-assisted hypothesis:** The writer criticizes disruptive parents, not the business. The bartender is praised for handling the situation responsibly. The model may be treating negative language about customers as negative sentiment toward the bar.

**Proposed future training/validation study:** Add labeled third-party complaint examples to training and compare validation errors where criticism targets people other than the business.

#### 7. yelp:test:22807 - Ambiguous review update

Official label 1; prediction 0; P(positive) 0.000042; 60 processed tokens before truncation.

Exact evidence:

```text
They really did change the service up
```

```text
Horrible service.
```

```text
disrespect your paying customers
```

**AI-assisted hypothesis:** The opening edit mentions changed service, but the older body remains strongly negative. The edit does not clearly say whether service improved. A later rating or missing context could explain the positive label; neither is confirmed.

**Proposed future training/validation study:** Label update direction in a separate training/validation sample and test whether retaining explicit update markers improves validation macro-F1.

#### 8. yelp:test:264 - Restaurant versus listing complaint

Official label 1; prediction 0; P(positive) 0.000089; 28 processed tokens before truncation.

Exact evidence:

```text
I like Applebee's
```

```text
something Yelp does
```

```text
But not cool!
```

**AI-assisted hypothesis:** The writer likes the restaurant but criticizes an inaccurate delivery listing that may come from Yelp. The negative prediction may reflect the listing complaint instead of the restaurant opinion. The intended label target is not fully clear.

**Proposed future training/validation study:** Add training examples separating platform complaints from restaurant opinions and measure validation errors on that distinction.

#### 9. yelp:test:10845 - Hyperbole and unclear overall rating

Official label 1; prediction 0; P(positive) 0.000114; 13 processed tokens before truncation.

Exact evidence:

```text
really pissed
```

```text
fresh baked pecan chocolate chip cookies
```

```text
someone needs to be shot for this.
```

**AI-assisted hypothesis:** The writer strongly complains about discontinued cookies. This also suggests liking the cookies, but the text does not state an overall positive judgment. Hyperbole or missing rating context might explain the positive label.

**Proposed future training/validation study:** Use independently labeled hyperbolic training examples, then compare macro-F1 on a fixed validation subset with similar figurative complaints.

#### 10. yelp:test:26683 - Concession and negated low rating

Official label 1; prediction 0; P(positive) 0.000116; 17 processed tokens before truncation.

Exact evidence:

```text
despite still not digging their ordering process
```

```text
their food is just too good
```

```text
disrespect with a 2 star review
```

**AI-assisted hypothesis:** The writer dislikes the ordering process but says the food is too good for a low rating. Negative words appear inside a positive conclusion. Negation is already retained in preprocessing, so removing negation is not the explanation.

**Proposed future training/validation study:** Add training-only contrast pairs with concessive and negated-rating phrases, then evaluate a separate validation set of these constructions.

### Near-threshold errors

#### 11. yelp:test:27612 - Mixed aspects near the threshold

Official label 0; prediction 1; P(positive) 0.500191; 44 processed tokens before truncation.

Exact evidence:

```text
Terrible experience this time!
```

```text
Very disappointed.
```

```text
'slice of bacon from heaven' was truly from heaven
```

**AI-assisted hypothesis:** The meal and service are criticized, with one strong exception for the bacon. The positive probability is almost 0.5, indicating an uncertain decision between conflicting aspects rather than a confident reading of the whole review.

**Proposed future training/validation study:** Compare mean and max pooling on the same BiLSTM, selecting by validation macro-F1 for mixed reviews without changing the threshold from test errors.

#### 12. yelp:test:28852 - Short idiomatic praise

Official label 1; prediction 0; P(positive) 0.499729; 5 processed tokens before truncation.

Exact evidence:

```text
Try it it's like mom used to make
```

**AI-assisted hypothesis:** The recommendation praises the food through a home-cooking comparison. Very few processed words provide context, and the probability is just below 0.5. The model may not consistently recognize this type of indirect praise.

**Proposed future training/validation study:** Add training-only paraphrases of short idiomatic recommendations and compare macro-F1 on the unchanged short-review validation slice.

#### 13. yelp:test:28068 - Colloquial mixed sentiment

Official label 0; prediction 1; P(positive) 0.500644; 39 processed tokens before truncation.

Exact evidence:

```text
The inside is amazing!!!!
```

```text
@#$%& them for their cover
```

```text
For a reason.
```

**AI-assisted hypothesis:** Praise for the interior and drinks is mixed with criticism of the cover charge, payment arrangements and crowd. The slightly positive prediction may miss the overall negative balance. Removed punctuation could affect emphasis, but this is untested.

**Proposed future training/validation study:** Add labeled colloquial mixed reviews to training and compare validation macro-F1 while keeping preprocessing and the decision threshold fixed.

#### 14. yelp:test:23782 - Improvement across visits

Official label 1; prediction 0; P(positive) 0.498702; 121 processed tokens before truncation.

Exact evidence:

```text
the last two have been exceptional.
```

```text
bland and unremarkable.
```

```text
I'd happily recommend Lobby's
```

**AI-assisted hypothesis:** An initially bad sandwich experience is followed by two excellent burger visits and a recommendation. The complete processed review fits the input limit. The model may not give enough weight to the more recent positive experiences.

**Proposed future training/validation study:** Add training contrast pairs in which a later visit changes the earlier opinion, then evaluate a separate temporal-update validation slice.

#### 15. yelp:test:19527 - Restrained praise near the threshold

Official label 1; prediction 0; P(positive) 0.498508; 17 processed tokens before truncation.

Exact evidence:

```text
Simple, well prepared food.
```

```text
Good atmosphere, good service.
```

```text
Heavier and more traditional
```

**AI-assisted hypothesis:** The review praises food, atmosphere and service, then makes a neutral comparison with another meal. The probability is just below 0.5. The descriptive words may be difficult to balance against the short positive statements.

**Proposed future training/validation study:** Add positive training examples containing neutral comparisons and measure macro-F1 on a fixed short-review validation slice.

### Long-review slice errors

#### 16. yelp:test:10081 - Long positive guide with warnings

Official label 1; prediction 0; P(positive) 0.039683; 581 processed tokens before truncation.

Exact evidence:

```text
I don't have a single complaint
```

```text
You can still get a quality room
```

```text
everybody is happy
```

**AI-assisted hypothesis:** The writer recommends the hotel but includes many warnings about rooms, queues and expectations. The review exceeds the input limit. Those warnings and omitted later context may contribute, although the opening already states satisfaction.

**Proposed future training/validation study:** Compare the current prefix with an equal-size head-and-tail input and evaluate the existing long-review validation slice.

#### 17. yelp:test:4664 - Long review with a worsening second visit

Official label 0; prediction 1; P(positive) 0.930167; 570 processed tokens before truncation.

Exact evidence:

```text
Great food, cheap beer, attentive service
```

```text
Zero stars.
```

```text
you'll want to go somewhere else.
```

**AI-assisted hypothesis:** A very positive first visit is followed by poor service and a negative final recommendation. The review exceeds the input limit, so the later verdict may be partly omitted while much of the early praise remains.

**Proposed future training/validation study:** Compare 384-token prefix and head-and-tail inputs under the same training settings, reporting overall and long-review validation macro-F1.

#### 18. yelp:test:9911 - Long mixed review with a positive ending

Official label 1; prediction 0; P(positive) 0.002201; 563 processed tokens before truncation.

Exact evidence:

```text
The tables here are cursed
```

```text
These people really made our trip memorable
```

```text
I'm giving this place 4 stars
```

**AI-assisted hypothesis:** The writer lists several complaints but later praises staff, housekeeping and room service, ending with four stars. The review exceeds the input limit. Missing some later positive context is one possible cause of the negative prediction.

**Proposed future training/validation study:** Test an equal-length head-and-tail representation against the prefix, using validation macro-F1 for long reviews to select the approach.

#### 19. yelp:test:9919 - Long review with a service-based reversal

Official label 1; prediction 0; P(positive) 0.037934; 536 processed tokens before truncation.

Exact evidence:

```text
Do I have complaints?  PLENTY.
```

```text
great treatment of dumbass me merited another star
```

```text
THAT'S SERVICE.
```

**AI-assisted hypothesis:** Hotel complaints are offset by appreciation for help after an injury. Some later detail may be truncated, but the first sentence already mentions the higher rating. Both context loss and mixed-sentiment weighting could matter.

**Proposed future training/validation study:** Compare head-and-tail and prefix inputs on the same training split, then inspect the predefined long-review validation results.

#### 20. yelp:test:26052 - Ironic setup followed by a recommendation

Official label 1; prediction 0; P(positive) 0.035674; 461 processed tokens before truncation.

Exact evidence:

```text
we need another burger joint, like earth needs global warming
```

```text
Zinburger's best, by far
```

```text
i'm giving it to Zin
```

**AI-assisted hypothesis:** A long poem criticizes the number of burger restaurants before recommending this particular one. The review exceeds the input limit and uses unusual wording. The ironic setup and possible loss of the final recommendation may both contribute.

**Proposed future training/validation study:** Compare equal-length prefix and head-and-tail inputs on training/validation data, with a separately labeled long ironic-review validation slice.

## dilated_cnn

Training source: `reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/training/dilated_cnn`. Selected checkpoint SHA-256: `f3883a005a369ef495c99bebe3cd51c4a7858c3f58121b6dbcfecee341dd8ac2`.

Recorded reviewed flags: 0 / 20. Every explanation below remains labeled as an AI-assisted draft.

### Confident false positives

#### 1. yelp:test:9938 - Mixed value and hygiene comments

Official label 0; prediction 1; P(positive) 0.999979; 113 processed tokens before truncation.

Exact evidence:

```text
you can't beat it for the price and location
```

```text
the drains in the shower and the sink backed up
```

**AI-assisted hypothesis:** The writer likes the price and location and would probably return, but also describes backed-up drains and dirty shower water. The positive prediction may reflect the favorable overall wording while giving too little weight to the hygiene complaint.

**Proposed future training/validation study:** Add reviewed mixed-aspect training examples and compare validation macro-F1 on reviews combining value praise with hygiene complaints.

#### 2. yelp:test:5752 - Negative update above an older positive review

Official label 0; prediction 1; P(positive) 0.999972; 260 processed tokens before truncation.

Exact evidence:

```text
This was a 4-star review
```

```text
customer service have gone down the tubes
```

**AI-assisted hypothesis:** The opening note says food and service have deteriorated, but most of the review preserves earlier praise. The prediction may follow that longer positive account rather than the updated judgment. The whole processed review fits in the input.

**Proposed future training/validation study:** Train a version that marks review updates explicitly; compare validation macro-F1 on updated reviews against the current preprocessing.

#### 3. yelp:test:29330 - Positive wording with a negative supplied label

Official label 0; prediction 1; P(positive) 0.999968; 23 processed tokens before truncation.

Exact evidence:

```text
Wow love the place
```

```text
Great place to come and relax
```

**AI-assisted hypothesis:** The short review praises the cleanliness and atmosphere and recommends visiting. There is no visible complaint explaining the supplied negative label. Missing rating context or an ambiguous label could account for the mismatch; the text cannot establish which.

**Proposed future training/validation study:** Audit similar short training reviews for text-label agreement; compare validation macro-F1 before and after documented training-only corrections.

#### 4. yelp:test:23815 - Bakery praise mixed with limited grocery appeal

Official label 0; prediction 1; P(positive) 0.999880; 129 processed tokens before truncation.

Exact evidence:

```text
The bakery here is tops
```

```text
The deli isn't the greatest
```

**AI-assisted hypothesis:** The writer strongly praises bakery products while describing the store as basic and the deli as unremarkable. The positive prediction fits much of the text, but conflicts with the supplied label. The intended overall rating is unclear from these mixed comments.

**Proposed future training/validation study:** Add reviewed training examples with conflicting product and store opinions; compare macro-F1 on a matching validation slice.

#### 5. yelp:test:17407 - Good first visit followed by disappointment

Official label 0; prediction 1; P(positive) 0.999878; 453 processed tokens before truncation.

Exact evidence:

```text
it will probably be the last
```

```text
Bad place for someone
```

**AI-assisted hypothesis:** The writer recalls an excellent first visit, then describes smaller portions and disappointment after a menu change. The final negative comparison falls beyond the 384-token limit. Losing that ending may contribute, although the retained opening also says this will probably be the last visit.

**Proposed future training/validation study:** Train with a 288-token beginning and 96-token ending; compare long-review validation macro-F1 with the current 384-token prefix.

### Confident false negatives

#### 6. yelp:test:22807 - Brief service update above an older complaint

Official label 1; prediction 0; P(positive) 0.000012; 60 processed tokens before truncation.

Exact evidence:

```text
They really did change the service up
```

```text
Horrible service.
```

**AI-assisted hypothesis:** The opening edit says service changed, while the remaining text describes rude treatment during an earlier visit. That detailed complaint may dominate the prediction. The edit does not explicitly describe the new service quality, leaving some uncertainty about the positive label.

**Proposed future training/validation study:** Preserve an explicit edit marker during training and compare macro-F1 on validation reviews containing later updates.

#### 7. yelp:test:30793 - Past complaints and improved current ownership

Official label 1; prediction 0; P(positive) 0.000301; 64 processed tokens before truncation.

Exact evidence:

```text
This place is so much better since they changed owners.
```

```text
It was horrible.
```

```text
Now its much better.
```

**AI-assisted hypothesis:** The writer clearly separates poor service under previous owners from a much better current experience. The negative prediction may mix these two periods instead of following the current recommendation. Both the complaint and the positive ending fit in the input.

**Proposed future training/validation study:** Add reviewed ownership-change training examples; compare validation macro-F1 on reviews contrasting past and current experiences.

#### 8. yelp:test:26683 - Praise expressed by rejecting a low rating

Official label 1; prediction 0; P(positive) 0.000408; 17 processed tokens before truncation.

Exact evidence:

```text
their food is just too good to disrespect with a 2 star review
```

**AI-assisted hypothesis:** The writer dislikes the ordering process but says the food is too good to deserve a two-star review. The negative prediction may reflect the complaint and low-rating phrase without capturing that the writer is rejecting that low rating.

**Proposed future training/validation study:** Add label-checked training examples that reject low ratings, then compare macro-F1 on a corresponding validation slice.

#### 9. yelp:test:21215 - Criticism of patrons while supporting staff

Official label 1; prediction 0; P(positive) 0.000730; 100 processed tokens before truncation.

Exact evidence:

```text
the bartender was very cordial
```

```text
had mind to do the right thing
```

**AI-assisted hypothesis:** The writer condemns the parents and their unsupervised children, then praises the bartender for handling the situation. The negative prediction may confuse criticism of other customers with criticism of the business. The closing approval is present in the input.

**Proposed future training/validation study:** Add reviewed training examples distinguishing customer behavior from staff behavior; compare validation macro-F1 on that distinction.

#### 10. yelp:test:10845 - Missing favorite product with unclear overall rating

Official label 1; prediction 0; P(positive) 0.001060; 13 processed tokens before truncation.

Exact evidence:

```text
they don't have the fresh baked pecan chocolate chip cookies anymore
```

**AI-assisted hypothesis:** The writer angrily complains that a particular cookie is no longer available. This suggests appreciation for the product but gives little direct praise for the current business experience. The supplied positive label is difficult to explain from this short text alone.

**Proposed future training/validation study:** Audit comparable training reviews for product-versus-business sentiment; compare validation macro-F1 after documented training-only label-quality improvements.

### Near-threshold errors

#### 11. yelp:test:11322 - Mixed satisfaction close to the decision threshold

Official label 1; prediction 0; P(positive) 0.499721; 116 processed tokens before truncation.

Exact evidence:

```text
The food is consistently good
```

```text
can't seem to give them five stars
```

**AI-assisted hypothesis:** The review praises consistent food, service and atmosphere but says the restaurant is not especially memorable. Its positive probability is just below 0.5. The error reflects a close decision on mixed sentiment, rather than an obvious missing complaint or recommendation.

**Proposed future training/validation study:** Add reviewed mixed-sentiment training examples and compare validation macro-F1 while keeping the decision threshold fixed at 0.5.

#### 12. yelp:test:2116 - Polite wording around a missed appointment

Official label 0; prediction 1; P(positive) 0.500325; 22 processed tokens before truncation.

Exact evidence:

```text
would've shown up
```

```text
wait around thinking
```

**AI-assisted hypothesis:** The writer says an estimate would have been appreciated if the worker had arrived, then complains about wasted time. The wording expresses a missed appointment, not gratitude. The barely positive prediction may miss that distinction in the conditional phrasing.

**Proposed future training/validation study:** Add label-checked missed-appointment examples with polite wording to training; compare macro-F1 on a matching validation slice.

#### 13. yelp:test:17408 - Mall complaints mixed with praise for an alternative

Official label 0; prediction 1; P(positive) 0.500393; 192 processed tokens before truncation.

Exact evidence:

```text
WHO BUILDS AN OUTDOOR MALL IN THE MIDDLE OF THE FREAKIN' DESERT!
```

```text
the best part of my trip to this mall was leaving
```

**AI-assisted hypothesis:** The review criticizes the outdoor heat, confusing layout and poorly equipped play area. Positive wording describes expectations, isolated amenities and a different store visited afterward. The slightly positive prediction may combine these details without following the negative assessment of the mall.

**Proposed future training/validation study:** Add reviewed training examples that praise alternatives while criticizing the reviewed business; compare validation macro-F1 on similar contrasts.

#### 14. yelp:test:19294 - Sarcasm expressed through apparently positive phrases

Official label 0; prediction 1; P(positive) 0.502173; 49 processed tokens before truncation.

Exact evidence:

```text
If you like bad chinese food, you'll love this place.
```

```text
Have fun tonight, indeed.
```

**AI-assisted hypothesis:** The opening says people who like bad food will love this restaurant. Later, a fortune-cookie message is repeated after a stomach complaint. These apparently positive phrases are sarcastic in context, which may help explain the slightly positive prediction.

**Proposed future training/validation study:** Add label-checked sarcastic praise examples to training and compare macro-F1 on a separate validation sarcasm slice.

#### 15. yelp:test:13231 - Good facilities outweighed by a poor viewing experience

Official label 0; prediction 1; P(positive) 0.502511; 124 processed tokens before truncation.

Exact evidence:

```text
There are plenty of downsides, though.
```

```text
There are plenty of better megaplexes around here.
```

**AI-assisted hypothesis:** The writer likes the seating and prices but criticizes noise, staff inaction and distracting light. The ending recommends other theaters. The positive prediction may give too much weight to the favorable opening relative to the complaints and final recommendation.

**Proposed future training/validation study:** Add reviewed facility-versus-service contrasts to training; compare macro-F1 on matching validation reviews and the full validation set.

### Long-review slice errors

#### 16. yelp:test:4664 - Positive first visit and an omitted negative ending

Official label 0; prediction 1; P(positive) 0.994477; 570 processed tokens before truncation.

Exact evidence:

```text
two completely different restaurants
```

```text
Zero stars.
```

```text
go somewhere else.
```

**AI-assisted hypothesis:** The first visit receives enthusiastic praise, while the second involves long waits and poor service. The zero-star judgment and advice to go elsewhere fall beyond the 384-token limit. Missing that conclusion may contribute, although several complaints remain visible.

**Proposed future training/validation study:** Train with a 288-token beginning and 96-token ending; compare long-review validation macro-F1 with prefix-only input.

#### 17. yelp:test:9911 - Late praise for staff falls outside the input

Official label 1; prediction 0; P(positive) 0.004710; 563 processed tokens before truncation.

Exact evidence:

```text
brand new penthouse
```

```text
These people really made our trip memorable
```

**AI-assisted hypothesis:** The review includes complaints about prices, dining and staff, followed by praise for a manager arranging a better room. That resolution and the memorable-trip conclusion fall beyond the input limit. Their omission may make the retained review appear more negative.

**Proposed future training/validation study:** Train with a 288-token beginning and 96-token ending; compare validation macro-F1 on long reviews describing a resolved problem.

#### 18. yelp:test:37318 - Anticipation and staff praise within a negative review

Official label 0; prediction 1; P(positive) 0.788689; 561 processed tokens before truncation.

Exact evidence:

```text
i was super excited
```

```text
YU SHOuld not go to this place.
```

```text
The only thing elevated about this place is the price.
```

**AI-assisted hypothesis:** The writer praises an imagined meal and some staff, then describes disappointing food, prices and management. The input loses the final price criticism and recommendation. Earlier complaints are still present, so omitted context and mixed opinions are both plausible contributors.

**Proposed future training/validation study:** Compare beginning-plus-ending and prefix-only training at the same token budget, using macro-F1 on long validation reviews.

#### 19. yelp:test:29374 - Food praise alongside worsening service complaints

Official label 0; prediction 1; P(positive) 0.613981; 547 processed tokens before truncation.

Exact evidence:

```text
Even the best food and atmosphere can taste horrendous
```

```text
There ARE better AYCE in town
```

**AI-assisted hypothesis:** The writer likes some dishes but describes delays, mishandled food and a dismissive manager. The closing recommendation of better alternatives falls outside the input limit. Losing that ending may weaken the negative judgment, though substantial complaints remain in the retained text.

**Proposed future training/validation study:** Train with a 256-token beginning and 128-token ending; compare long-review validation macro-F1 against the existing prefix-only model.

#### 20. yelp:test:9919 - Hotel complaints followed by praise for emergency help

Official label 1; prediction 0; P(positive) 0.030189; 536 processed tokens before truncation.

Exact evidence:

```text
their great treatment of dumbass me merited another star
```

```text
they were all VERY nice to me.
```

```text
THAT'S SERVICE.
```

**AI-assisted hypothesis:** The writer lists hotel problems but raises the rating because staff helped after an injury. Most details of that help and the final service praise exceed the input limit. The opening already mentions the higher rating, so truncation is only a possible contributor.

**Proposed future training/validation study:** Train with a 256-token beginning and 128-token ending; compare validation macro-F1 on long reviews containing later staff praise.

## Student and team review

Review each full packet, retained input, official label and selected-model prediction. Verify or revise the hypotheses in personal words and propose a testable study. Set a human-reviewed flag only after that review occurs. This rendering does not add, remove or approve human annotations.

Independent teammate evidence, the joint comparison/report, individual viva understanding and the remaining whole-lab obligations require their own completion evidence. The root AI_USE.md disclosure describes assistance; these drafts do not establish independent authorship.
