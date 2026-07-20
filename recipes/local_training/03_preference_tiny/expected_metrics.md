# Expected signals

Beta sets the numerical scale of DPO's loss. The sigmoid argument is
`beta * (log pi_theta - log pi_ref)`, evaluated as the difference
between the chosen and rejected continuations, so at `beta=0.1` the loss
moves on a small numerical scale even when the underlying log-probability
margins have shifted substantially. The step-to-step jitter this produces
reflects that small scale, not optimization instability, and chasing it
with a smaller learning rate usually hurts more than it helps. Beta is
also the strength of the KL pull toward the reference: a larger beta
holds the policy closer to SFT, a smaller one lets it drift further.

The reference-using algorithms (DPO, IPO, KTO) print both `Trainable:`
and `Frozen reference:` at startup. They hold a second frozen copy of the
model, so their resident weights are roughly double the reference-free
variants. The reference is scored under a no-gradient pass, so the added
cost is weights, not activations. SimPO, ORPO, CPO, and RePO drop the
reference forward and replace it with a length- or margin-based
regularizer. The savings show up directly in
`estimate_vram.py`, which is the reason the reference-free variants
are the default choice on memory-constrained machines.

The recipe samples from three prompts (`What makes a good friend?`,
`How do I learn to cook?`, `Tell me about dogs.`) chosen to differ from
the SFT recipe's three, so the qualitative read is not just measuring
memorization of the SFT prompts.
Preference tuning shifts which coherent answer the policy prefers among
the answers the base already assigns nontrivial mass to. How coherent
any of those answers is in absolute terms is fixed by the base.

These runs validate the loss path and the reference-model handling.
What they do not do is move preference behavior far, and the reason is
built into the scale. HH-RLHF preferences at this scale mostly reflect
stylistic surface features that a 7.5M-parameter model can fit, and the
updates stay small enough that the policy remains close to SFT no matter
how many steps the trainer takes.
