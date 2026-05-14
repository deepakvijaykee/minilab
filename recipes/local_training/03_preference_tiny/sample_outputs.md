# Sample output

```text
Estimated VRAM (rough planning estimate)
  ...
Trainable: checkpoints/local_training/sft/step_500 (gpt, 7,512,832 params)
Frozen reference: checkpoints/local_training/sft/step_500
hh: 1000 examples for dpo
  step 50 loss=...
  ...
  saved checkpoints/local_training/preference_dpo/step_300
  wrote checkpoints/local_training/preference_dpo/step_300/run_metrics.json

--- After DPO ---
  Q: What makes a good friend?
  A: ...
```

For reference-free algorithms (SimPO, ORPO, CPO, RePO) the `Frozen reference`
line is absent and the save directory becomes `preference_<algorithm>`.
