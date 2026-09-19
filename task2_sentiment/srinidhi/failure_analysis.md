# BiLSTM failure analysis - AI-assisted draft

**Pending student review.** These annotations are an AI-assisted reading of all 20 complete review texts. The CSV's original `error_type` and `testable_fix` fields remain blank, and `student_reviewed` remains `False`. Drafts are stored only in the three `ai_draft_*` columns.

The frozen BiLSTM checkpoint has 1,516 errors among 38,000 test reviews (accuracy 96.01%, macro-F1 0.9601). This deliberately selected sample contains five confident false positives (FP), five confident false negatives (FN), five errors nearest the 0.5 threshold, and five distinct long-review errors. It is not a random sample and cannot establish error-type prevalence.

The types below are hypotheses grounded in wording, not diagnosed model mechanisms. Six reviews exceed the 384-token input cap; cutoff observations use the actual tokenizer. Suggested fixes are proposals for a future study using training/validation data and fresh held-out evaluation. No label changes, tuning, or retraining were performed from these test errors.

| Review ID | Selection group | Draft error type | Short evidence | Future testable fix |
|---|---|---|---|---|
| yelp:test:29330 | Confident FP | Possible text-label disagreement | "love the place" and "worth a try"; supplied label is negative. | Audit text-rating agreement independently; retain the test label. |
| yelp:test:22663 | Confident FP | Mixed aspects and sarcastic price criticism | "Food Fantastic" contrasts with "more for show than content" and "go somewhere else." | Test aspect-aware aggregation on conflicting food/value reviews. |
| yelp:test:17407 | Confident FP | Past-versus-current sentiment and omitted conclusion | First visit: "fantastic"; omitted ending: "SOOOOOO TINY" and "Bad place." | Compare head-and-tail or chunk aggregation on long validation reviews. |
| yelp:test:9706 | Confident FP | Conditional praise and conflicting hotel aspects | "beautiful and spacious" versus "if someone else was paying" and a more casual next choice. | Test sentence aggregation on conditional-recommendation validation cases. |
| yelp:test:4655 | Confident FP | Mixed pros/cons with possible sarcasm | "bad: the parking and the crowd"; "totally the coolest place, like, ever." | Test punctuation-preserving input on validation pros/cons and sarcasm cases. |
| yelp:test:22807 | Confident FN | Edit-versus-original text ambiguity | "EDIT: They really did change the service" precedes "Horrible service." | Test update-aware weighting; separately audit ambiguous rating context. |
| yelp:test:10845 | Confident FN | Missing-product complaint with ambiguous overall sentiment | "really pissed" that the cookies are unavailable; supplied label is positive. | Evaluate missing-favorite complaints separately, with human aspect judgments. |
| yelp:test:15892 | Confident FN | Opposing gambling and lodging sentiment | "love the environment" but "don't stay here"; room and cleanliness complaints. | Compare aspect aggregation for entertainment versus lodging validation cases. |
| yelp:test:30793 | Confident FN | Past-versus-current sentiment reversal | "old owners ... terrible" versus "Now its much better" and "new owners." | Evaluate before/now contrast handling on independent validation examples. |
| yelp:test:32999 | Confident FN | Literal versus negative lexical sense | "vacuums can REALLY suck" followed by "Thank goodness." | Test contextual contrast pairs for literal versus evaluative word senses. |
| yelp:test:21910 | Near threshold | Very short pragmatic dismissal | "one word ... next!"; positive probability 0.5004. | Test punctuation retention on a short-utterance validation slice. |
| yelp:test:4504 | Near threshold | Mixed concessions and scoped negative recommendation | "food wasn't bad" but "Can't really recommend" after a 45-minute wait. | Test negation-scope contrasts and recommendation weighting on validation. |
| yelp:test:29329 | Near threshold | Current praise versus nostalgic price complaint | "Good wings" contrasts with missing the old 10-cent price. | Separate current quality from past-price comparisons in validation analysis. |
| yelp:test:34560 | Near threshold | Implicit amenity-based positive sentiment | Describes useful amenities and a "great location"; positive probability 0.4983. | Evaluate descriptive positive reviews and sentence-level aggregation. |
| yelp:test:8115 | Near threshold | Contrastive idiom weakened by preprocessing | "far from good" becomes "far good" under the recorded stopword removal. | Test function-word retention on an independent idiom validation slice. |
| yelp:test:10081 | Long review | Conditional advice and mixed hotel sentiment | "don't have a single complaint" alongside warnings about poor room choices. | Test hypothetical-warning distinctions; separately compare long-review chunks. |
| yelp:test:4664 | Long review | Visit-level reversal with omitted final verdict | First visit: "Great food"; omitted later verdict: "Zero stars" and go elsewhere. | Compare temporal aggregation and head-and-tail coverage on validation. |
| yelp:test:9911 | Long review | Mixed aspects with omitted positive resolution | Omitted ending praises helpful management and explicitly gives "4 stars." | Test coverage of late overall judgments using validation head/tail or chunks. |
| yelp:test:37318 | Long review | Imagined praise versus actual dissatisfaction | Imagined "brothy goodness" precedes "just terrible" food; negative summary is omitted. | Test expectation/experience contrasts and independently assess tail coverage. |
| yelp:test:26587 | Long review | Past-versus-present and food-versus-service conflict | "soured my experience"; the missing order and long-wait details fall after the cutoff. | Test temporal/aspect aggregation and tail coverage on validation data. |

Sources: [annotated review CSV](outputs/full/bilstm/required_20_errors_for_review.csv), [frozen BiLSTM metrics](outputs/full/bilstm/metrics.json). Each CSV row retains the original label, prediction, probability, full text, and checkpoint SHA256.
