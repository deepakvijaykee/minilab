# Expected Signals

- SFT training loss should remain finite and usually decline.
- The final checkpoint should answer the printed instruction prompts in a more
  instruction-like format than the base LM.
- Short default runs may still produce weak answers; increase pretraining and
  SFT steps before judging quality.
