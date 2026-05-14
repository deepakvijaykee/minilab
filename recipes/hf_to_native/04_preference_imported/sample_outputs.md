# Sample output

```text
Trainable: checkpoints/imported/smollm2-135m-sft/step_100 (gpt, 134,515,008 params)
hh: 200 examples for simpo
  step 25 loss=...
  ...
  saved checkpoints/imported/smollm2-135m-simpo/step_50
  wrote checkpoints/imported/smollm2-135m-simpo/step_50/run_metrics.json

--- After SIMPO ---
  Q: What makes a good friend?
  A: ...
```

For `ALGORITHM=dpo` the script prints an extra `Frozen reference: <path>`
line above the dataset line and the save directory becomes
`smollm2-135m-dpo`.
