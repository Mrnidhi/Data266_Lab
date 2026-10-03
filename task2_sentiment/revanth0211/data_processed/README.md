# Processed data

No raw or processed Yelp review copy is committed. The notebook downloads
`fancyzhx/yelp_polarity`, shuffles with seed 3143, and selects 100,000
training, 10,000 validation, and 20,000 test rows.

Text is lowercased and tokenized with the notebook's saved regular expression.
The vocabulary is fit on training text only, capped at 40,000 entries, and
stored in [../outputs/vocab.json](../outputs/vocab.json). Sequences are padded
or truncated to 256 tokens. The notebook records class and word-length
distributions under `../outputs/`.
