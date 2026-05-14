# Expected signals

- Output checkpoint is `checkpoints/imported/<model>-<algorithm>/step_<MAX_STEPS>`,
  default `checkpoints/imported/smollm2-135m-simpo/step_50`.
- `run_metrics.json` is written alongside.
- For `dpo`, `ipo`, or `kto` the script prints both `Trainable:` and
  `Frozen reference:` lines and memory roughly doubles versus SimPO.
- 50 steps over 200 examples is below what is needed for a meaningful
  preference shift on a 135M base. It is enough to confirm the loss runs
  finite and the checkpoint round-trips through `model.save` / `load`.

If SimPO loss climbs steadily, the most common cause is that the SFT
checkpoint at `step_100` is too undertrained; rerun recipe 03 with more
steps before retrying.
