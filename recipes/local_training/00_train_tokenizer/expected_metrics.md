# Expected signals

The script prints three lines, and reading them in order gives a clean
sanity check that the tokenizer behaves correctly.

The first line reads `Corpus: <N> chars from <N> TinyStories rows` and
confirms that the dataset loaded. An empty corpus almost always
means the Hugging Face datasets cache failed to populate, usually because
of a network problem or a stale lock file under `.cache/huggingface/`.

The second line reads `Saved <path> (vocab=<N>)` and reports the trained
vocabulary size. BPE can settle slightly below the requested size when the
corpus runs out of distinct frequent pairs to merge. A
4k vocabulary on 5000 TinyStories rows usually saturates close to the
request. A 16k vocabulary on the same corpus generally will not, because
there are simply not enough merge candidates in such a small slice of
text. If you need the full 16k, the fix is more rows, not a different
setting: the realized size just reflects how much distinct structure the
corpus actually contains.

The third line reads `"Once upon a time there was a little girl named
Lily." -> <N> tokens, roundtrip OK`. This is the encode and decode
consistency check on a fixed sample sentence. WordPiece prints
`decodes as "..."` instead of `roundtrip OK` because its detokenizer is
not a strict inverse of the encoder by construction, so an exact
round-trip is not something the scheme promises in the first place.
