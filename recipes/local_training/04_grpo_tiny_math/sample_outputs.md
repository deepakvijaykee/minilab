# Sample output

```text
Estimated VRAM (rough planning estimate)
  ...
Trainable: checkpoints/local_training/sft/step_500 (gpt, 7,512,832 params)
Frozen reference: checkpoints/local_training/sft/step_500
GSM8K: train=500 test=50
  step 50 loss=...
  ...
  saved checkpoints/local_training/grpo/step_100
  wrote checkpoints/local_training/grpo/step_100/run_metrics.json

--- After GRPO (held-out GSM8K test) ---
  Q: Natalia sold clips to 48 of her friends in April ...
  A: ...  (predicted=48, expected=72, WRONG)

GSM8K test subset (50 of full split) accuracy: 3/50 = 6.0%
```

DAPO does not print the `Frozen reference` line because it has no
reference model to load. Setting `EVAL_EXAMPLES=0` changes the summary
line to `GSM8K test accuracy: <correct>/<total> = <pct>%` and runs the
full GSM8K test split rather than a fixed subset, which produces a
tighter estimate at the cost of a longer eval pass.
