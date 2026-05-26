# 00 Train tokenizer

This recipe trains a small BPE tokenizer on TinyStories. Every downstream
recipe in `local_training/` loads the resulting `tokenizer.json`, so the
choice made here propagates through pretraining, SFT, preference
optimization, and RLVR. Vocabulary size is the only meaningful decision
the recipe exposes, but it is consequential, because the embedding matrix
dominates the non-attention parameter count at the laptop scale.

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
```

By default the recipe builds a 4096-token BPE vocabulary over 5000
TinyStories rows and writes the result to
`checkpoints/local_training/tokenizer.json`. Downstream recipes read that
path unless `TOKENIZER` is overridden.

The 4k choice is what keeps `gpt-10m` actually around ten million
parameters on a laptop. At `dim=256` a 4096-token embedding matrix is
roughly one million parameters, which leaves the transformer blocks to
account for the rest of the budget. Pushing the vocabulary up to 16k
moves the same preset toward roughly thirteen million parameters, almost
entirely through the embedding and output projection. The preset table
in the root README covers that range.
