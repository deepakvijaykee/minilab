# Sample output

```text
Loaded checkpoints/imported/smollm2-135m (gpt, 134,515,008 params)
Alpaca: 500 examples
  step 50 loss=...
  ...
  saved checkpoints/imported/smollm2-135m-sft/step_100
  wrote checkpoints/imported/smollm2-135m-sft/step_100/run_metrics.json

--- After SFT ---
  Q: Give three tips for staying healthy.
  A: ...
```

The parameter count above is for SmolLM2-135M. For `smollm2-360m` it
is around 360M, with the rest of the block shape identical because
the trainer logging is shared across model sizes.
