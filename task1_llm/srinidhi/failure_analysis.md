# Srinidhi — Character GPT failure analysis

These observations use the actual epoch-12 checkpoint and the zero-based entries in `outputs/full/generations.json`. This is an AI-assisted analysis draft for student review; it does not claim that the student has already reviewed the model.

## 1. Repetition: generation 4, greedy

> He wanted to go outside and play with his friends. He wanted to go outside and play with his friends.

The same sentence appears twice consecutively and then returns again later. The story stops developing Tom's original goal of helping his friend. Greedy decoding repeatedly picks the most likely next character, which can reinforce a common phrase. The limited model and context may also contribute; this run does not isolate the cause.

A useful next experiment is to compare sampling and a small repetition penalty using the same prompts and checkpoint. Measure repeated 4-grams and inspect whether the story still makes sense. A lower repetition score alone does not establish better writing.

## 2. Broken grammar: generation 7, sampled

> The dog chased it on, and touched it broke.

The clauses do not form a grammatical description of an action, and the object of the action is unclear. This occurs within the generated continuation, not at the imposed length cutoff. Character-level predictions can spell common words while failing to combine them into a meaningful sentence. Sampling adds variation but can also select a less plausible continuation.

A useful next experiment is to reduce the temperature from 0.8 to 0.6 while keeping top-k at 40, then compare several fixed prompts for grammar and repetition. That change has not been tested here.

## 3. Character identity drift: generation 5, sampled

> His friend looked at him and said, "I don't know, Lily. I will be happy. I will be very excited."

The prompt names Tom, but the dialogue addresses Lily without introducing her. The text loses track of the established character. A 256-character attention window limits accessible context, and next-character prediction does not explicitly enforce character consistency. These are plausible explanations rather than demonstrated causal findings.

A useful next experiment is to compare a longer context at a similar training budget and manually track names across the same prompts. Keep this separate from a decoding-only experiment so the source of any improvement is clearer.

## What the metrics do and do not show

Validation cross-entropy improved to 0.7197 and character perplexity to 2.0538, but those next-character metrics do not guarantee coherent stories. Across the five fixed prompts, the repeated 4-gram fraction is 0.1956 with greedy decoding and 0.0207 with sampling. Sampled text has more variety, yet the grammar and identity examples show remaining weaknesses. This is a small qualitative sample, not an estimate of the failure rate over all stories.

Every continuation is capped at 256 emitted characters. A trailing partial word or unfinished sentence at that boundary is an evaluation length limit and is not counted as a separate model failure here. The reported negative train-validation loss gap also compares training with dropout against validation without dropout; it should not be interpreted as a matched estimate of generalization.
