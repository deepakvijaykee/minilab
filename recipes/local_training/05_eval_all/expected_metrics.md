# Expected signals

For each checkpoint, `evaluate.py` prints a small fixed block:

- `Loaded <path> (<model_name>) on <device> (<N> params)`
- `<dataset> validation perplexity: <ppl>` (plus
  `text8 validation bits/char` on text8)
- `Distinct-1`, `Distinct-2`, `Distinct-3`, and `Self-BLEU-4` over ten
  sampled generations
- Five truncated samples under `--- Samples ---`

What to read is not any single block in isolation but the trajectory
across `base -> sft -> preference -> grpo`. Three patterns recur along
that trajectory, and each reflects the intended distributional shift.

Perplexity on TinyStories almost always rises after SFT, even though
SFT is nominally training the model. The evaluation distribution stays
TinyStories while SFT is dragging the policy toward Alpaca, which
means the run is trading eval-domain perplexity for response-domain
capability. The same shift, in the same direction and for the same
reason, happens again after preference tuning.

Distinct-N drops from `base` through `sft` to `preference`, and
Self-BLEU-4 rises in step, because each alignment step narrows the
output distribution onto a more specific mode of the data. Preference
tuning is usually the most aggressive narrower at this scale, since it
explicitly suppresses the rejected mode rather than merely upweighting a
target one. A well-behaved `grpo` run then partially reverses the
narrowing, because the verifier reward does not collapse onto a single
canonical answer the way preference labels can.

If Distinct-N falls to zero, meaning the model is emitting a single
repeated string, the run has hit mode collapse. The usual culprit is
preference tuning with beta too small, which loosens the KL pull toward
SFT enough that the policy can park on a single high-margin response. RLVR rarely collapses in the same way at this
scale because the verifier reward is hard to game with one fixed
string: only strings that hit the exact numeric answer score, and the
model cannot retrieve those reliably without learning the underlying
task.

`Skipping <label>: missing <path>` lines record which stages have run so
far. They are how the recipe evaluates a partial pipeline cleanly and
leaves the stages that have not run for a later pass.
