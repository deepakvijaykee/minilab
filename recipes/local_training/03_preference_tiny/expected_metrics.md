# Expected signals

DPO's loss is bounded in scale by beta. The sigmoid argument is
`beta * (log pi_theta - log pi_ref)` evaluated as a difference between
the chosen and rejected continuations, so with `beta=0.1` the implicit
trust region around the SFT policy is tight and the loss moves on a
small numerical scale even when the underlying log-probability margins
have shifted substantially. The step-to-step jitter that this produces
reflects that scale mismatch rather than optimization instability, and
chasing it with a smaller learning rate generally hurts more than it
helps.

The reference-using algorithms (DPO, IPO, KTO) print both `Trainable:`
and `Frozen reference:` at startup, and their activation memory is
roughly double the reference-free variants because chosen and rejected
each forward through both the policy and the reference. SimPO, ORPO,
CPO, and RePO drop the reference forward and replace it with a length-
or margin-based regularizer. The savings show up directly in
`estimate_vram.py`, which is the reason the reference-free variants
are the default choice on memory-constrained machines.

The recipe samples from three prompts (`What makes a good friend?`,
`How do I learn to cook?`, `Tell me about dogs.`) that are deliberately
different from the SFT recipe's three. The point is that the
qualitative read is not measuring memorization of SFT prompts.
Preference tuning shifts which coherent answer the policy prefers among
the answers the base already assigns nontrivial mass to; how coherent
any of those answers is in absolute terms is fixed by the base.

These runs validate the loss path and the reference-model bookkeeping.
They are not a preference benchmark, and reading them as one would
overinterpret the result. HH-RLHF preferences at this scale mostly
track stylistic surface features that a seven million parameter model
can fit, and the beta-scaled trust region keeps the policy close to
SFT regardless of how many steps the trainer takes.
