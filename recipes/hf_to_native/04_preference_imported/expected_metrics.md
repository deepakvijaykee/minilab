# Expected Metrics

This recipe writes a native preference checkpoint and `run_metrics.json`.

Default output:

```text
checkpoints/imported/smollm2-135m-simpo/step_50
```

For `dpo`, `ipo`, or `kto`, the script uses the SFT checkpoint as the frozen
reference through Minilab's existing reference-checkpoint path.
