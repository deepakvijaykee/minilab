# Expected signals

The script prints three lines of state:

1. `Corpus: <N> chars from <N> TinyStories rows`. This is the sanity check
   that the dataset actually loaded. An empty corpus usually means the HF
   datasets cache failed.
2. `Saved <path> (vocab=<N>)`. The tokenizer was written and reports the
   trained vocabulary size. For BPE this can be slightly smaller than the
   requested `--vocab-size` if the corpus runs out of merges.
3. `"Once upon a time there was a little girl named Lily." -> <N> tokens,
   roundtrip OK`. The sample sentence encodes and decodes back exactly.
   WordPiece prints the decoded text instead of asserting roundtrip,
   because WordPiece is lossy.

These are correctness gates; tokenizer quality at this scale is downstream
of how well later pretraining picks up the cadence.
