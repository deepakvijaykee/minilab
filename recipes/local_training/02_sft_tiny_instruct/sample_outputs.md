# Sample output

```text
Estimated VRAM (rough planning estimate)
  ...
Loaded checkpoints/local_training/lm/step_1000 (gpt, 7,512,832 params)
Alpaca: 2000 examples
  step 100 loss=...
  ...
  saved checkpoints/local_training/sft/step_500
  wrote checkpoints/local_training/sft/step_500/run_metrics.json

--- After SFT ---
  Q: Give three tips for staying healthy.
  A: ...

  Q: What is the capital of France?
  A: ...

  Q: Explain gravity.
  A: ...
```

The exact answer text is not stable across seeds; the "Q/A" shape is.
