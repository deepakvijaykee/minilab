# Expected signals

The script prints three lines:

1. `Corpus: <N> chars from <N> TinyStories rows`. Sanity check that the
   dataset loaded. An empty corpus usually means the HF datasets cache
   failed.
2. `Saved <path> (vocab=<N>)`. The trained vocabulary size. BPE can land
   slightly below the requested size when the corpus runs out of distinct
   frequent pairs to merge. A 4k vocab on 5000 TinyStories rows usually
   saturates; a 16k vocab on the same corpus will not. If you need a
   large vocab, train on a larger corpus or accept the smaller realized
   size.
3. `"Once upon a time there was a little girl named Lily." -> <N> tokens,
   roundtrip OK`. The sample sentence encodes and decodes losslessly.
   WordPiece prints `decodes as "..."` instead, because its detokenizer
   is not a strict inverse. That is a property of the WordPiece scheme,
   not a bug in this implementation.
