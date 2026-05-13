# 01 Pretrain Tiny GPT

Pretrain a tiny GPT-style LM on TinyStories using one of the official Minilab
presets.

```bash
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
```

Default preset: `gpt-10m`.

Useful overrides:

```bash
PRESET=gpt-25m MAX_STEPS=3000 bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
```
