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

DAPO omits the `Frozen reference` line. With `EVAL_EXAMPLES=0` the summary
becomes `GSM8K test accuracy: <correct>/<total> = <pct>%` over the full
test split.
