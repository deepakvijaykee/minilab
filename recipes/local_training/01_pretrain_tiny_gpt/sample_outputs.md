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

The reported parameter count is for the default 4096-token BPE
tokenizer. With a larger vocabulary the embedding and output projection
grow proportionally and the count moves with them, which is why a
16k-vocabulary `gpt-10m` lands closer to thirteen million.
