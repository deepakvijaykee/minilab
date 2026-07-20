# Expected signals

The output checkpoint lives at
`checkpoints/imported/<model>-<algorithm>/step_<MAX_STEPS>`, which
under defaults resolves to
`checkpoints/imported/smollm2-135m-simpo/step_50`. The
`run_metrics.json` file is written alongside.

For `dpo`, `ipo`, or `kto`, the script prints an additional
`Frozen reference:` line at startup, and the resident model weights
roughly double: a second frozen copy of the model is held alongside the
trainable policy. The reference is scored under a no-gradient pass, so
the added cost is weights rather than backprop activations. On a 135M
base that second copy is the difference between fitting on 8GB of VRAM
and not, which is the whole reason SimPO is the default here.

Fifty steps over 200 pairs is below what would be needed to shift
preference behavior in any quantitatively meaningful way on a 135M
base. The margin will move on most pairs, but the policy barely moves
from the SFT checkpoint. What this run verifies is therefore not a
preference result but three structural properties: that the loss path
is finite and stable, that the optimizer state allocates correctly, and
that the native preference trainers accept imported weights without
ad-hoc modification.

A SimPO loss that climbs steadily rather than descending almost
always points back to recipe 03: the SFT checkpoint at `step_100` is
too undertrained to provide a useful starting distribution. The fix
is to rerun recipe 03 with more steps before retrying preference
tuning here.
