# Processed data

No raw or processed TinyStories copy is committed. The notebook records the
deterministic seed, 40-million-character source-pool limit, 90/10 disjoint
story split, character vocabulary, and window-sampling logic. It downloads
`roneneldan/TinyStories` through Hugging Face Datasets when local text files
are unavailable.

The final experiment uses 100,000 training windows and 10,000 validation
windows, each 256 characters long. Windows never cross story boundaries.
