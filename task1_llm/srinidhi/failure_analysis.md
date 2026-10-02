# Part 1: three observed generation failures

These excerpts come from the selected epoch-16 checkpoint. Generation IDs are
zero-based entries in `outputs/full/generations.json`. This is AI-assisted
analysis; student review remains pending. Quotes are checked against the saved
continuations. The examples illustrate errors, not their frequency in all stories.

## Case 1 — repeated sentence

Generation ID: 8

Decoding: greedy, temperature 0, seed 3382.
Prompt: `The little girl learned that sharing`

```text
She learned that sharing is important and that sharing is important. She learned that sharing is important and that sharing is important.
```

Failure type: repetition.

Observation: the same sentence repeats without adding an event or explanation.
This occurs before the output cutoff. Choosing the most likely next character
can keep returning to a familiar phrase, although this example does not prove
which model or decoding choice caused the loop.

Possible improvement: compare a lower-repetition decoding setting on fixed
validation prompts and check that the story still makes sense. Do not select
settings using the final examples alone.

## Case 2 — missing object in a sentence

Generation ID: 5

Decoding: sampled, temperature 0.8, top-k 40, seed 3363.
Prompt: `Tom wanted to help his friend, but`

```text
He tried to test his friend's first, but it wasn't ugly at first.
```

Failure type: broken grammar and unclear reference.

Observation: `his friend's first` does not identify what Tom is testing, and
`it` has no clear object to refer to. This is a complete saved sentence, so
its missing meaning is not caused by the 256-character cutoff. The model
combines common word forms without maintaining a complete sentence structure.

Possible improvement: check the same grammatical pattern across validation
samples while varying temperature; count errors as well as repetition.

## Case 3 — unexplained change of intention

Generation ID: 1

Decoding: sampled, temperature 0.8, top-k 40, seed 3343.
Prompt: `Once upon a time, a little bird`

```text
Tweety wanted to help the bird.

The bird said, "Hello, bird! What can I do with you?" Tweety said, "I want to bite the bird."
```

Failure type: inconsistent character intention.

Observation: Tweety switches from helping the bird to wanting to bite it,
without an event explaining the change. The quoted sentences finish before
the cutoff. Good next-character prediction does not directly enforce stable
character goals across sentences.

Possible improvement: review goal consistency on fixed validation stories
alongside cross-entropy before changing the training or decoding recipe.

Each of the five prompts has a greedy and a sampled continuation, capped at
256 characters. Trailing fragments caused by that cap are not counted as
additional errors. Repeated 4-gram rates are 0.1952 for greedy decoding and
0.0124 for sampling. Lower validation loss and fewer repeated phrases do not
by themselves establish coherent stories. The proposed improvements above
are suggestions, not additional experiments claimed by this submission.
