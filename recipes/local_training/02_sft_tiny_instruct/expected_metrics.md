# Expected signals

- SFT loss drops fast, then plateaus low. The base already knows the
  language; SFT is re-weighting the head's output distribution toward the
  response template. Most of the gradient signal is in the first
  100-200 steps, after which the trainer is mainly fine-tuning rare-token
  probabilities.
- The default Alpaca prompts (`Give three tips for staying healthy.`,
  `What is the capital of France?`, `Explain gravity.`) are deliberately
  broad-knowledge. A 7M model has nowhere near the capacity to memorize
  that factual content, so expect on-topic but wrong answers. That is
  the response template winning over the content head, which is the right
  ordering of effects at this scale.
- `run_metrics.json` lands under `checkpoints/local_training/sft/`.

If the answers come out as raw TinyStories text (kids, dogs, "once upon
a time"), the prompt loss mask is not active or the loaded checkpoint is
being trained from scratch. Check that `--checkpoint` resolved to the
pretrained step and that the trainer is `SFTTrainer`, not the bare
language-model trainer.
