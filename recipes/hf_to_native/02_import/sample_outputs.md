# Sample output

```text
Imported HuggingFaceTB/SmolLM2-135M -> checkpoints/imported/smollm2-135m
Tokenizer: checkpoints/imported/smollm2-135m/tokenizer.json
Checkpoint: checkpoints/imported/smollm2-135m
Logit check: max_abs_diff=4.768e-06 mean_abs_diff=2.146e-07
```

The `Logit check` line only prints with `VERIFY=1` (the default). For
non-Llama presets the script raises a `require()` from `_native_config`
before reaching the print block.
