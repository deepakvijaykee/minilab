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

For the reference-free algorithms (SimPO, ORPO, CPO, RePO) the
`Frozen reference` line is absent, because the trainer never loads a
second copy of the policy, and the save directory is named after the
algorithm rather than carrying the `preference_dpo` suffix.
