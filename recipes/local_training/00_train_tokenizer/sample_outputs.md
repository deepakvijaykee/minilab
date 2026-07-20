# Sample output

```text
Corpus: 4,712,083 chars from 5000 TinyStories rows
[BPE training progress ...]
Saved checkpoints/local_training/tokenizer.json (vocab=4096)
  "Once upon a time there was a little girl named Lily." -> 14 tokens, roundtrip OK
```

The exact character count and token count vary with the dataset slice
and the tokenizer family. The shape of the three lines is what matters
for sanity. WordPiece is the only family whose third line
differs: it prints `decodes as "..."` in place of `roundtrip OK`, since
its detokenizer is not built to invert the encoder exactly.
