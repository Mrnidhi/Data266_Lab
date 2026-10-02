# Frozen Yelp Polarity preprocessing

`full/` contains selected JSONL reviews/manifests from `fancyzhx/yelp_polarity`,
resolved revision `bbf1c97a1f0cf005e5aded43839fd814654a1557`. Seed 2342
stratifies 560,000 official training rows into 504,000 train and 56,000
validation rows; all 38,000 official test rows are retained. Each split is
balanced. The October 1 desktop cache matches historical row contents and
ordering exactly. Root `verification/part2_desktop_data_identity.json` records
actual CRLF-byte and normalized-LF hashes; encoded features are in `full_encoded/`.

Tokenization casefolds, expands contractions, removes HTML/punctuation and a
frozen customized stopword list, preserving negation/contrast words. A
40,000-word dictionary with minimum frequency two is fitted on training only.
Each model learns its own embeddings; the first 384 tokens form the input.
Stemming/lemmatization is omitted. Empty processed text maps to UNK: 22 train,
one validation and zero test rows. There are zero malformed or blank raw rows.
Duplicate texts are audited and retained; seven text hashes occur in train
and test. Official test IDs/labels/texts are never silently changed or deleted.

Distribution, truncation/OOV and duplicate evidence is in `../outputs/full/`.
JSONL/encoded caches are ignored by Git but included in the Part 2 ZIP; a
clone needs public-data preparation or a verified cache before full reproduction.
Preserve complete manifests and hashes when moving data. The tuner verifies
all split/cache fingerprints, including test metadata/arrays, then removes
test tensors before training. Only train/validation affect weights or selection.

See [finalization guide](../../../PART2_FINALIZATION.md) for preparation,
training, replay and package verification. Other members must keep their
preprocessing/model artifacts in their own folders.
