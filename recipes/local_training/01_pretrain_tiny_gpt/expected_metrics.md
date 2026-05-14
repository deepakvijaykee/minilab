# Expected signals

- The estimator prints rough VRAM before training; if your GPU has less
  headroom than the reported peak, drop `BATCH_SIZE` or `SEQ_LEN`
  before the run.
- Initial loss on a 4k-vocab model with uniform predictions is
  `log(4096) ~= 8.3`. Default 1000-step runs usually land in the 5-6 range:
  the easy entropy is gone, the model is climbing the long tail of
  bigram structure. If you are still above 6 at the end, the most likely
  cause is a vocab mismatch with the loaded `tokenizer.json`.
- Sample quality at 1000 steps looks like fluent tokens without coherent
  narrative. The model has the unigram and short-range bigram distribution
  but not the story templates. Coherence is roughly a `params x steps`
  threshold and appears on TinyStories around `gpt-25m x 3000`.
- The trainer logs every 100 steps, evaluates every 500. TinyStories
  has a held-out split so `Eval perplexity` prints at the end; OpenWebText
  is the only dataset that skips eval (streaming, no fixed split).
- `run_metrics.json` is written to the final `step_<N>` directory and
  also to the recipe save root.
