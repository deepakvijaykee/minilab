# Sample output

```text
== base: checkpoints/local_training/lm/step_1000 ==
Loaded checkpoints/local_training/lm/step_1000 (gpt) on cuda (7,512,832 params)
tinystories validation perplexity: 38.4
Distinct-1: 0.412
Distinct-2: 0.731
Distinct-3: 0.853
Self-BLEU-4: 0.214

--- Samples ---
  Once upon a time there was a little girl ...
  ...

== sft: checkpoints/local_training/sft/step_500 ==
...

Skipping grpo: missing checkpoints/local_training/grpo/step_100
```

The exact numbers depend on dataset slice and seed; the per-label block
shape is stable.
