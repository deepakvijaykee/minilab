# Expected signals

- Output checkpoint is `checkpoints/imported/<model>-<algorithm>/step_<MAX_STEPS>`,
  default `checkpoints/imported/smollm2-135m-simpo/step_50`.
- `run_metrics.json` is written alongside.
- For `dpo`, `ipo`, or `kto` the script prints `Trainable:` and
  `Frozen reference:`, and activation memory roughly doubles. On a 135M
  imported model that is the difference between fitting on 8GB and not;
  the default switching to SimPO is on purpose.
- 50 steps over 200 pairs is below what is needed for a real preference
  shift on a 135M base. The margin will move on most pairs, but the
  policy is still close to SFT in KL. The point of this recipe is the
  loss path, the optimizer state shape, and that the native preference
  trainers accept imported weights cleanly.

If SimPO loss climbs steadily, the most likely cause is that the SFT
checkpoint at `step_100` is too undertrained to give a useful starting
distribution. Rerun recipe 03 with more steps before retrying.
