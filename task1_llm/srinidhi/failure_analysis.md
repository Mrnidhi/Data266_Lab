# Srinidhi — Part 1 observed generation failures

These cases come from the newly trained desktop RTX 5090 model, selected at
epoch 12 using validation cross-entropy. Generation IDs are zero-based entries
in `outputs/full/generations.json`. This is an AI-assisted analysis for student
review; it does not claim that the student has personally reviewed or defended
the interpretations. Every quoted snippet is verified against the actual saved
continuation.

## Case 1 — repeated sentence

Generation ID: 2

Decoding: greedy, temperature 0, seed 3352. Prompt: `Lily found a small red box. Inside`

```text
She wanted to play with it and see what it was. She wanted to play with it and see what it was.
```

Failure type: repetition.

Observation: the identical sentence appears twice consecutively, adding no new
event or explanation of what Lily finds. This occurs inside the continuation,
before the output-length boundary.

Possible explanation: repeatedly choosing the most likely next character can
reinforce a common phrase. The model's learned distribution and limited context
may also contribute; this run does not isolate their individual effects.

Testable next step: compare greedy decoding with temperature 0.6 and 0.8 at
top-k 40 on the same checkpoint and prompts. Record repeated 4-grams and inspect
coherence, since fewer repetitions alone do not establish better stories.

## Case 2 — incorrect verb form

Generation ID: 7

Decoding: sampled, temperature 0.8, top-k 40, seed 3373.
Prompt: `One rainy day, the dog and the cat`

```text
They saw a loud rainbow and stopped and wanted to saw a park.
```

Failure type: broken grammar.

Observation: after `wanted to`, the model produces the past-tense form `saw`
instead of the base form `see`. The sentence is complete within the saved
continuation, so the grammar error is not caused by truncation. `Loud rainbow`
also shows a questionable combination of sensory descriptions.

Possible explanation: accurate character predictions and familiar word spellings
do not enforce sentence-level grammatical constraints. Temperature sampling
can admit less probable constructions, but one sample cannot demonstrate that
sampling caused the error.

Testable next step: lower temperature to 0.6 while keeping the checkpoint,
top-k, prompts and sampling seeds fixed. Count this verb-form error across the
same sample set and check whether repetition increases.

## Case 3 — incoherent event and object references

Generation ID: 9

Decoding: sampled, temperature 0.8, top-k 40, seed 3383.
Prompt: `The little girl learned that sharing`

```text
She heard a loud noise. It was made of many flours. She took the barn and took a delicious learnt and put it on the boat.
```

Failure type: semantic incoherence.

Observation: `It` appears to refer to the noise, but describing that noise as
made of flours is unexplained. The next action moves from a barn to a boat and
treats `learnt` as a food-like object, without introducing a coherent event.
The text contains familiar words while losing their relationships. These
sentences precede the character cutoff.

Possible explanation: a next-character objective rewards local prediction and
does not explicitly require consistent entities or plausible event links.
Limited model capacity and context are possible contributors, but no controlled
ablation here establishes a cause.

Testable next step: compare a longer context with the current 256-character
context in a separately controlled experiment, holding prompts and decoding
fixed and reporting compute differences. Review event and entity consistency
alongside cross-entropy; a better next-character score alone is insufficient.

The five prompts produce ten continuations, each capped at 256 characters.
Trailing partial words at that imposed boundary are not counted as additional
model failures. Across these continuations, repeated 4-gram rate is 0.2482 for
greedy decoding and 0.0281 for sampling. These are small-sample observations,
not estimated failure rates for all TinyStories. Validation perplexity 2.0567
and next-character accuracy 77.31% therefore coexist with poor story coherence.

The generalization gap compares validation without dropout against online
training with dropout and changing weights. Its negative value should not be
interpreted as a matched evaluation of both splits. The proposed fixes above
have not been tested in this run.
