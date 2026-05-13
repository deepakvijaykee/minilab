# Expected Signals

- Each available checkpoint prints validation perplexity and diversity metrics.
- Missing checkpoints are skipped instead of failing the whole evaluation pass.
- Samples should make qualitative regressions visible across base, SFT,
  preference, and RLVR checkpoints.

Use this recipe to build a measured table after you have run the full track on a
specific GPU.
