# Expected signals

Per checkpoint, `evaluate.py` prints:

- `Loaded <path> (<model_name>) on <device> (<N> params)`
- `<dataset> validation perplexity: <ppl>` (and `text8 validation bits/char:
  <bpc>` if `--dataset text8`).
- `Distinct-1`, `Distinct-2`, `Distinct-3`, `Self-BLEU-4` over 10 sampled
  generations (the recipe sets `NUM_SAMPLES=10`).
- Five truncated samples under `--- Samples ---`.

Reading across labels:

- Perplexity usually goes down from `base` to `sft` only if the eval set
  matches the SFT distribution; on TinyStories eval it can rise after SFT
  because the model has drifted toward Alpaca cadence.
- Distinct-N usually drops from `base` to `sft` to `preference` (the model
  becomes more peaked). Self-BLEU-4 moves the opposite way.

The recipe prints `Skipping <label>: missing <path>` for any checkpoint
directory that does not exist; that line is the audit trail of which
stages have completed.
