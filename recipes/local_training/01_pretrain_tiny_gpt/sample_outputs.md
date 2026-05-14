# Sample output

```text
Estimated VRAM (rough planning estimate)
  ...
Data: tinystories train=10000 eval=2000
GPT: 7,512,832 params
  step 100 loss=...
  ...
  saved checkpoints/local_training/lm/step_1000
  wrote checkpoints/local_training/lm/step_1000/run_metrics.json

Eval perplexity: 38.4
  once upon a time there was a little girl ...
  the little dog ...
  she was very happy ...
```

Param counts depend on tokenizer vocabulary; the value above is for the
default 4096-vocab BPE tokenizer.
