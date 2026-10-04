# Member-specific Part 1 preprocessing

This member's full TinyStories cache is under `full/gpt_<fingerprint>/`, with
`train.jsonl`, `validation.jsonl` and `manifest.json`. It contains 100,000
training stories and 10,000 validation stories selected using seed 2342.
The manifest freezes source revision, original row identifiers, text hashes,
split counts and selection policy; cache loading verifies the content hashes.
Data are not mixed with another member's processed outputs.

`lab1.gpt.prepare_data(config)` regenerates the configured cache from the public
dataset. `lab1.gpt.data_cache_paths(config)` returns its precise relative paths.
Full training creates train-only character mappings and fixed-length,
story-bounded input/target windows at runtime. The run's `vocabulary.json`
records PAD/UNK/BOS/EOS and the ordered training characters; the notebook
constructs and displays its own `char_to_idx` and `idx_to_char` dictionaries.
Window coverage
and padding behavior are documented in the member README and tests.

The selected run records its manifest and vocabulary under
`reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part1/depth_context_full/`.
It uses exactly the same frozen story split and vocabulary as the preserved
October 1 baseline. Its frozen
story cache and character mappings are included in `reproducibility/packages/parts1-2-20261002/Part1_Srinidhi_2342.zip` for
portable offline reproduction. Large data remain excluded from Git; a clone
alone obtains documentation and metadata, not the cached story files. Raw
Hugging Face downloads are under `task1_llm/data/huggingface/` and need not be
copied if the verified frozen member cache is available.

After `git lfs pull`, run `python scripts/prepare_srinidhi.py --part 1` from the
repository root to restore this exact cache before executing the notebook.
