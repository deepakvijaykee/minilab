# 01 Pretrain tiny GPT

This recipe pretrains a small GPT on TinyStories and produces the base
checkpoint that recipes 02 through 05 build on. Pretraining at this
scale is doing one specific thing: fitting a next-token distribution
over a narrow, child-grammar corpus. The downstream alignment stages can
only re-weight what this base already assigns probability to, which makes
the quality of this checkpoint a primary determinant of everything that
follows.

```bash
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
```

The defaults are `--preset gpt-10m`, `--seq-len 512`, `--batch-size 8`,
`--max-steps 1000`, and `--max-examples 10000`, with gradient
checkpointing on. The `run.sh` wrapper first invokes
`scripts/estimate_vram.py` so that obvious memory pressure is caught
before the optimizer is constructed rather than several minutes into
training.

A thousand steps is enough to produce loss curves that look right and
samples that already carry the TinyStories cadence, but it is not enough
for coherent stories. Anything that needs to be judged qualitatively
should run for at least three thousand steps and use `gpt-25m`:

```bash
PRESET=gpt-25m MAX_STEPS=3000 bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
```

The preset chosen here flows through every later recipe via the saved
`config.json`, so picking a larger preset at pretraining time is how you
scale the entire downstream pipeline. Switching presets
mid-pipeline is not supported because the trainers refuse to load a
checkpoint whose model dimensions disagree with the active config.
