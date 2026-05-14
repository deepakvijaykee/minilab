# 00 Train tokenizer

Trains a small BPE tokenizer on TinyStories. Every downstream recipe in
`local_training/` loads the resulting `tokenizer.json`, so this is always the
first step.

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
```

It builds a 4k-vocab BPE tokenizer over 5000 TinyStories rows and writes to
`checkpoints/local_training/tokenizer.json`. Downstream recipes read that
path unless `TOKENIZER` is overridden.

A 4k vocab keeps the embedding matrix tiny (about 1M params at `dim=256`),
which is what makes `gpt-10m` actually 10M-ish on a laptop. Bumping the vocab
to 16k pushes the same preset toward ~13M parameters; the README preset table
covers that range.
