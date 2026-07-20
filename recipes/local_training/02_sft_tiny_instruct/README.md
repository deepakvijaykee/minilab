# 02 SFT tiny instruct

This recipe runs supervised fine-tuning on Alpaca, starting from the
pretrained checkpoint produced by recipe 01. The point of SFT at this
scale is not to teach the model new facts, which it has nowhere near
the capacity to absorb, but to shift the head's output distribution
toward an instruction-following template so that the downstream
preference and RLVR stages have a sensible starting policy.

```bash
bash recipes/local_training/02_sft_tiny_instruct/run.sh
```

By default the recipe loads `checkpoints/local_training/lm/step_1000`
and trains for 500 steps with `batch_size=4`, `lr=1e-4`, over 2000
Alpaca rows. The output checkpoint lands under
`checkpoints/local_training/sft`.

The SFT trainer masks loss on prompt tokens and supervises only the
response tokens, so the base language model's text habits carry over
unchanged while the answer template is the only thing being relearned.
With a thousand-step base, 500 steps of SFT, and 2000 Alpaca rows, the
model begins producing answer-shaped continuations rather than the
TinyStories drift the base alone would emit. Genuine instruction quality
is a separate axis that needs a stronger base. What moves at this scale
is the shape of the answer, not its content, and that ordering, format
before content, is the most instructive thing this stage shows about the
alignment stack.
