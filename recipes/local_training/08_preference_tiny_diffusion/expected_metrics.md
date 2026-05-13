# Expected Signals

- Diffusion DPO or VRPO loss should stay finite.
- The SFT checkpoint is also loaded as the frozen reference.
- Generated infill samples should not collapse into empty or repeated text.

Tiny preference runs validate the objective and checkpointing path. They are not
preference benchmark claims.
