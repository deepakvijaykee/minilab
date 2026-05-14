# Expected signals

- DPO's loss is bounded by `beta`: the sigmoid argument is
  `beta * (log pi_theta - log pi_ref)` differenced over chosen and rejected.
  With `beta=0.1` the implicit trust region is tight, so the loss moves on
  a small scale even when the policy itself moves a lot. Step-to-step
  jitter reflects that mismatch, not instability.
- Reference-using algorithms (DPO, IPO, KTO) print both `Trainable:` and
  `Frozen reference:` and roughly double activation memory: chosen and
  rejected each forward through both models. SimPO, ORPO, CPO, and RePO
  drop the reference forward; `estimate_vram.py` reflects the saving.
- Sample prompts (`What makes a good friend?`, `How do I learn to cook?`,
  `Tell me about dogs.`) are different from SFT's three so the qualitative
  read is not measuring memorization. Preference tuning shifts *which*
  coherent answer comes out; how coherent it is, is fixed by the base.

These runs validate the loss path and the reference-model glue. They are
not a preference benchmark. HH-RLHF preferences mostly track stylistic
surface features that a 7M model can fit, and the beta-scaled trust region
keeps the policy close to SFT regardless.
