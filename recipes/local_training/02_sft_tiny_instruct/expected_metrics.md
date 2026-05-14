# Expected signals

- SFT loss drops faster than pretraining loss because most of the work is
  adapting the response distribution, not learning the language.
- The end-of-run samples answer three fixed prompts:
  `Give three tips for staying healthy.`, `What is the capital of France?`,
  `Explain gravity.`. After a 1000-step base and 500 SFT steps the answers
  are short and on-topic but often factually wrong. That is the expected
  ceiling at this scale.
- `run_metrics.json` is written under `checkpoints/local_training/sft/`.

If the answers look like raw TinyStories text (kids, dogs, "once upon a
time"), the SFT loss masking is probably not active: confirm that the loaded
checkpoint is actually the pretrained one and not a from-scratch model.
