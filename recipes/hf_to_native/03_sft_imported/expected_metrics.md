# Expected Metrics

This is a short native SFT sanity run on an imported checkpoint.

- `checkpoints/imported/<model>-sft/run_metrics.json` should be written.
- The final checkpoint should be `checkpoints/imported/<model>-sft/step_<MAX_STEPS>`.
- On CUDA, run metrics include peak PyTorch allocator memory.
