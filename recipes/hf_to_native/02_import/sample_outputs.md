# Sample output

```text
Imported HuggingFaceTB/SmolLM2-135M -> checkpoints/imported/smollm2-135m
Tokenizer: checkpoints/imported/smollm2-135m/tokenizer.json
Checkpoint: checkpoints/imported/smollm2-135m
Logit check: max_abs_diff=4.768e-06 mean_abs_diff=2.146e-07
```

The `Logit check` line only prints with `VERIFY=1`, which is the
default in the run script. For a preset outside the supported set,
anything but Llama-compatible or dense Qwen3, the script raises a
`require()` from `_native_config` before it reaches the print block, so a
bad mapping never yields a checkpoint at all. That order matters because
logit drift from a wrong weight mapping does not always produce obviously
broken text in the sampler, so a refused import catches what visual
inspection of the samples would miss.
