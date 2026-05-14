# Expected signals

Per checkpoint, `evaluate.py` prints:

- `Loaded <path> (<model_name>) on <device> (<N> params)`
- `<dataset> validation perplexity: <ppl>` (plus
  `text8 validation bits/char` on text8)
- `Distinct-1`, `Distinct-2`, `Distinct-3`, `Self-BLEU-4` over 10 sampled
  generations
- Five truncated samples under `--- Samples ---`

The interesting read is the cross-stage trajectory:

- Perplexity on TinyStories almost always rises after SFT, even though
  SFT is "training." The eval distribution stays TinyStories while SFT
  drags the policy toward Alpaca; you trade eval-domain perplexity for
  response-domain capability. The same shift hits again after preference
  tuning.
- Distinct-N drops monotonically `base -> sft -> preference -> grpo` and
  Self-BLEU-4 rises symmetrically. Each alignment step narrows the output
  distribution. Preference tuning is the most aggressive narrower because
  it explicitly suppresses the rejected mode.
- If Distinct-N falls to zero (one repeated string) you have hit mode
  collapse, usually from preference tuning at too-large `beta`. RLVR
  rarely collapses the same way because the verifier reward is hard to
  game with a single fixed string.

`Skipping <label>: missing <path>` lines are the audit trail of which
stages completed.
