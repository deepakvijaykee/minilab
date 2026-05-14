# 02 SFT tiny instruct

Supervised fine-tuning on Alpaca. Loads the pretrained checkpoint from
recipe 01 and trains an instruction-following variant.

```bash
bash recipes/local_training/02_sft_tiny_instruct/run.sh
```

By default it consumes `checkpoints/local_training/lm/step_1000` and runs
for 500 steps at batch 4, `lr=1e-4`, over 2000 Alpaca rows. Output lands
in `checkpoints/local_training/sft`.

The SFT trainer masks loss on prompt tokens and only supervises responses,
so the base LM's text habits carry over while only the answer style is
re-learned. With a 1000-step base, 500 steps of SFT, and 2000 Alpaca rows,
the model will start producing answer-shaped continuations rather than
TinyStories drift, but full instruction quality requires a stronger base.
