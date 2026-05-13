# Expected Signals

- Preference loss should stay finite.
- DPO, IPO, and KTO load the SFT checkpoint as a frozen reference.
- SimPO, ORPO, CPO, and RePO avoid the reference model and should estimate
  lower memory in `scripts/estimate_vram.py`.
- Generated samples should not collapse into empty or repeated responses.

Tiny preference runs are mainly for checking loss mechanics and qualitative
behavior, not for claiming preference benchmark quality.
