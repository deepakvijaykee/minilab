# Expected signals

For each checkpoint, `evaluate.py` prints a small fixed block:

- `Loaded <path> (<model_name>) on <device> (<N> params)`
- `<dataset> validation perplexity: <ppl>` (plus
  `text8 validation bits/char` on text8)
- `Distinct-1`, `Distinct-2`, `Distinct-3`, and `Self-BLEU-4` over ten
  sampled generations
- Five truncated samples under `--- Samples ---`

The interesting reading is not any single block in isolation but the
trajectory across `base -> sft -> preference -> grpo`. Three patterns
recur and are worth stating explicitly because they look like
regressions until they are interpreted correctly.

Perplexity on TinyStories almost always rises after SFT, even though
SFT is nominally training the model. The evaluation distribution stays
TinyStories while SFT is dragging the policy toward Alpaca, which
means the run is trading eval-domain perplexity for response-domain
capability. The same shift, in the same direction and for the same
reason, happens again after preference tuning.

Distinct-N drops roughly monotonically along
`base -> sft -> preference -> grpo`, and Self-BLEU-4 rises in a
symmetric way. Each alignment step narrows the output distribution
because each step pushes mass onto a more specific mode of the data.
Preference tuning is usually the most aggressive narrower at this
scale because it explicitly suppresses the rejected mode rather than
merely upweighting a target one.

If Distinct-N falls to zero, meaning the model is emitting a single
repeated string, the run has hit mode collapse. The usual culprit is
preference tuning with beta too large, which makes the implicit trust
region around SFT loose enough that the policy can park on a single
high-margin response. RLVR rarely collapses in the same way at this
scale because the verifier reward is hard to game with one fixed
string: only strings that hit the exact numeric answer score, and the
model cannot retrieve those reliably without learning the underlying
task.

`Skipping <label>: missing <path>` lines are the audit trail of which
stages completed. They are not failures; they are how this recipe lets
a partial pipeline be evaluated cleanly.
