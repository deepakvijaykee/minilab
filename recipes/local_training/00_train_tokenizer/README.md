# 00 Train Tokenizer

Train a small BPE tokenizer on TinyStories for the laptop GPU track.

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
```

The tokenizer is saved to `checkpoints/local_training/tokenizer.json` by default.
Downstream recipes use that path unless `TOKENIZER` is overridden.
