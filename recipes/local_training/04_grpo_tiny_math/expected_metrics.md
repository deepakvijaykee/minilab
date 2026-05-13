# Expected Signals

- The run should complete with finite policy loss.
- The held-out GSM8K subset accuracy prints at the end.
- Very short defaults may show little or no accuracy gain. Increase
  pretraining, SFT, `MAX_STEPS`, and `NUM_GENERATIONS` before judging RLVR
  quality.
- If memory is tight, reduce `MAX_NEW_TOKENS`, keep `BATCH_SIZE=1`, or try
  `ALGORITHM=rloo`.
