# Sample output

```text
Estimated VRAM (rough planning estimate)
  ...
Trainable: checkpoints/local_training/diffusion_sft/step_500 (mdlm, 27,xxx,xxx params, schedule=cosine)
Frozen reference: checkpoints/local_training/diffusion_sft/step_500
GSM8K: train=500 test=50
  step 50 loss=...
  ...
  saved checkpoints/local_training/diffusion_grpo/step_100
  wrote checkpoints/local_training/diffusion_grpo/step_100/run_metrics.json

--- After Diffusion GRPO (held-out GSM8K test) ---
  Q: Natalia sold clips to 48 of her friends in April ...
  A: ...  (predicted=, expected=72, WRONG)

GSM8K test subset (50 of full split) accuracy: 0/50 = 0.0%
```

The `predicted=` blank in the example is realistic at the default scale:
the diffusion base often produces no extractable number, which the verifier
treats as wrong.
