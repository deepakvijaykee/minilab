# Expected signals

- The estimator prints rough VRAM before training; if your GPU has less
  headroom than the reported `peak` number, drop `BATCH_SIZE` or `SEQ_LEN`
  before the run.
- TinyStories has a held-out split, so `Eval perplexity` prints at the end.
  OpenWebText is the only dataset that skips this path (streaming, no eval
  split in `pretrain_lm.py`).
- The trainer logs loss every 100 steps and runs evaluation every 500.
- The script samples from three fixed prompts (`once upon a time`,
  `the little dog`, `she was very happy`) at the end of the run. After 1000
  steps the samples are TinyStories-shaped tokens, not coherent stories.
  Coherent narratives need ~3000 steps and a larger preset.
- `run_metrics.json` lands both in the final `step_<N>` directory and in the
  recipe save directory.

If training loss is still above ~6 by the end of the default run, the most
common cause is a vocab/tokenizer mismatch between this recipe and the
tokenizer it was pointed at.
