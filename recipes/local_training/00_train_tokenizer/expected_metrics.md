# Expected signals

The script prints three lines, and reading them in order gives a clean
sanity check that the tokenizer is healthy.

The first line reads `Corpus: <N> chars from <N> TinyStories rows` and
confirms that the dataset actually loaded. An empty corpus almost always
means the Hugging Face datasets cache failed to populate, usually because
of a network problem or a stale lock file under `.cache/huggingface/`.

The second line reads `Saved <path> (vocab=<N>)` and reports the trained
vocabulary size. BPE can land slightly below the requested size when the
corpus runs out of distinct frequent pairs to merge, which is benign. A
4k vocabulary on 5000 TinyStories rows usually saturates close to the
request. A 16k vocabulary on the same corpus generally will not, because
there are simply not enough merge candidates in such a small slice of
text. The right response is to train on more rows rather than to read
the smaller realized size as a defect.

The third line reads `"Once upon a time there was a little girl named
Lily." -> <N> tokens, roundtrip OK`. This is the encode and decode
consistency check on a fixed sample sentence. WordPiece prints
`decodes as "..."` instead of `roundtrip OK`, because its detokenizer is
not a strict inverse of the encoder by construction. That is a property
of the WordPiece scheme rather than a bug in the implementation.
