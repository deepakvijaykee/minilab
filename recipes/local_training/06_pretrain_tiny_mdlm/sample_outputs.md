# Sample output

```text
Estimated VRAM (rough planning estimate)
  ...
Data: tinystories train=10000 eval=2000
MDLM: 27,xxx,xxx params
  step 100 loss=...
  ...
  saved checkpoints/local_training/diffusion/step_1000
  wrote checkpoints/local_training/diffusion/step_1000/run_metrics.json

--- Samples ---
  ...
```

Param count varies with tokenizer vocabulary (the diffusion model embeds
`vocab_size + 1` to reserve the [MASK] token). For models that cannot do
unconditional reverse sampling, the `--- Samples ---` block prints
`skipped: model requires clean x_0 context for reverse scoring` instead.
