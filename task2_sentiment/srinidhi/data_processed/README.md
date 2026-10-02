# Frozen Yelp Polarity data

`full/` contains JSONL splits/manifests from `fancyzhx/yelp_polarity`, revision `bbf1c97a1f0cf005e5aded43839fd814654a1557`. Seed 2342 stratifies 560,000 official training rows into 504,000 training and 56,000 validation rows. All 38,000 official test rows are retained; each split is balanced. The new selected suite uses these same frozen rows.

`verification/part2_desktop_data_identity.json` records byte and normalized-LF hashes and verifies historical row contents/order. Encoded features are in `full_encoded/`; copies must retain complete manifests and matching files.

Preprocessing casefolds, expands contractions, removes HTML/punctuation and selected stopwords, and preserves negation/contrast. The vocabulary cap is 40,000 with minimum frequency 2, fitted on training only. Each model learns its own embedding; input uses the first 384 processed tokens. Stemming/lemmatization is omitted to retain word forms. Empty processed text maps to UNK: 22 training rows, one validation row and no test rows. There are no malformed or blank raw rows.

Duplicate texts are audited and retained: seven hashes occur in both training and test. Official test IDs, labels and texts remain unchanged. Distributions, truncation, OOV and duplicates are documented in [current outputs](../outputs/README.md).

The tuner verifies all split/cache fingerprints, including test metadata and arrays, then removes the test dataset before training. Only training/validation tensors reach training and checkpoint selection. Earlier test results were observed, so repeated evaluation is not a newly sealed holdout.

Git does not include dataset caches. Use the complete local data or finalized ZIP, or prepare the public dataset and verify the manifest. The ZIP includes real processed files. See the [finalization guide](../../../PART2_FINALIZATION.md); other members keep their own preprocessing/model artifacts in their own folders.
