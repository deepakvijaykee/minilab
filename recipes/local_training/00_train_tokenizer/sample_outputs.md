# Sample output

```text
Corpus: 4,712,083 chars from 5000 TinyStories rows
[BPE training progress ...]
Saved checkpoints/local_training/tokenizer.json (vocab=4096)
  "Once upon a time there was a little girl named Lily." -> 14 tokens, roundtrip OK
```

Numbers will vary with the dataset slice and tokenizer type; the shape is
what matters. WordPiece prints `decodes as "..."` instead of `roundtrip OK`
because its WordPiece detokenizer is not a strict inverse.
