# 00 Train tokenizer

This recipe trains a small BPE tokenizer on TinyStories. Every downstream
recipe in `local_training/` loads the resulting `tokenizer.json`, so the
choice made here propagates through pretraining, SFT, preference
optimization, and RLVR. Vocabulary size is the only meaningful decision
the recipe exposes, but it is consequential: the token embedding and
output projection are the main lever on total parameter count as the
vocabulary grows.

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
```

By default the recipe builds a 4096-token BPE vocabulary over 5000
TinyStories rows and writes the result to
`checkpoints/local_training/tokenizer.json`. Downstream recipes read that
path unless `TOKENIZER` is overridden.

The 4k choice is what keeps `gpt-10m` near its ~7.5M size; the preset
name is nominal, and the realized count depends on the vocabulary. At
`dim=256` the token embedding is roughly one million parameters, and the
transformer blocks account for most of the rest. Pushing the vocabulary up to 16k
moves the same preset toward roughly thirteen million parameters, almost
entirely through the embedding and output projection. The preset table
in the root README covers that range.
