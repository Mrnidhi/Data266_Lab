# Part 1 failure analysis

I reviewed the generated text in `outputs/sample.txt` and chose three places where the model's weaknesses are especially clear. These are copied directly from the saved output.

| Exact generated excerpt | Failure type | What I noticed | One testable change |
|---|---|---|---|
| "She could see what it was and she was very excited. The little girl was very excited." | Repetition | The second sentence repeats almost the same idea without adding anything to the story. This makes the writing feel mechanical instead of moving the scene forward. | Generate the same prompt with a no-repeat 3-gram rule and compare how often phrases are repeated. |
| "She spotted a new style of little pencils. The pencils were so spaceship" | Grammar and meaning break down | "So spaceship" does not work grammatically, and the jump from ordinary pencils to a spaceship idea is not explained. The model seems to have learned local word patterns without keeping a clear meaning across the sentence. | Lower the sampling temperature and compare the number of incomplete or nonsensical phrases in several generated samples. |
| "She offered to continue to the little girl to come ou" | Incomplete and confused sentence | The subject and action are difficult to follow, and the output stops in the middle of the word "out." This looks like both a language-coherence problem and a fixed-length generation cutoff. | Allow generation to continue until an end-of-sentence token after the minimum length, then check whether samples still end mid-word or mid-sentence. |

Overall, the model learned the general style of a simple children's story, but it struggled to maintain meaning over several sentences. The output also suggests that decoding settings matter: repetition control, a slightly more conservative temperature, and sentence-aware stopping could improve readability even without retraining the model.
