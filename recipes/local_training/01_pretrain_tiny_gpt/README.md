# 01 Pretrain tiny GPT

Pretrains a small GPT on TinyStories. This is the base checkpoint that
recipes 02 through 05 load.

```bash
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
```

Defaults: `--preset gpt-10m`, `--seq-len 512`, `--batch-size 8`,
`--max-steps 1000`, `--max-examples 10000`, gradient checkpointing on. The
run.sh also calls `scripts/estimate_vram.py` first so memory hits get caught
before training starts.

The default 1000 steps is enough to produce loss curves that look right and
samples that have TinyStories cadence; it is not enough for coherent stories.
For anything you would judge qualitatively, bump `MAX_STEPS` to 3000+ and
move to `PRESET=gpt-25m`:

```bash
PRESET=gpt-25m MAX_STEPS=3000 bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
```

The preset choice flows through to every later recipe via the saved
`config.json`, so switching presets here means later recipes will train a
correspondingly larger model.
