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

The parameter count moves with the tokenizer vocabulary because the
diffusion model embeds `vocab_size + 1` entries, with the extra slot
reserved for the `[MASK]` token. For model families that cannot
perform unconditional reverse sampling, the `--- Samples ---` block
prints `skipped: model requires clean x_0 context for reverse scoring`
in place of generated text, which signals correctly that the sampler
needs a clean prefix rather than that pretraining went wrong.
