# Sample output

```text
Trainable: checkpoints/imported/smollm2-135m-simpo/step_50 (gpt, 134,515,008 params)
Frozen reference: checkpoints/imported/smollm2-135m-simpo/step_50
GSM8K: train=100 test=20
  step 5 loss=...
  ...
  saved checkpoints/imported/smollm2-135m-grpo/step_25
  wrote checkpoints/imported/smollm2-135m-grpo/step_25/run_metrics.json

--- After GRPO (held-out GSM8K test) ---
  Q: Natalia sold clips to 48 of her friends in April ...
  A: ...  (predicted=24, expected=72, WRONG)

GSM8K test subset (20 of full split) accuracy: 1/20 = 5.0%
```

Numbers will vary widely seed to seed at this scale; a single-digit accuracy
at the default scale is the expected ballpark, not a useful signal.
