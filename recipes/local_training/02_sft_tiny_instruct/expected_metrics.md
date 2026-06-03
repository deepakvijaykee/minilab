# Expected signals

SFT loss drops quickly and then plateaus low. The base already speaks
the language, so what the trainer is actually doing is reweighting the
head's output distribution toward the response template. Most of the
useful gradient signal arrives in the first 100 to 200 steps, after
which the optimizer is mainly polishing rare-token probabilities and
the loss curve flattens.

The default Alpaca prompts (`Give three tips for staying healthy.`,
`What is the capital of France?`, `Explain gravity.`) ask for broad
factual knowledge, the kind a seven million parameter model has nowhere
near the capacity to memorize. The natural outcome is on-topic but
factually wrong answers. That is the response
template winning over the content head, and at this scale it is the
correct ordering of effects: format converges first because format
lives in a low-dimensional subspace of the head, while content would
require representations the base does not have.

The `run_metrics.json` file is written under
`checkpoints/local_training/sft/`, alongside the model artifacts. That
file is the place to look for actual measured memory and timing.

The clearest failure to watch for is the base text bleeding through. If
the answers come out as raw TinyStories text, with children, dogs, and
"once upon a time" appearing in response to Alpaca prompts, then either
the prompt loss
mask is not active or the loaded checkpoint is being trained from
scratch rather than fine-tuned. The two things to verify are that
`--checkpoint` resolved to the pretrained step and that the trainer
in use is `SFTTrainer` rather than the bare language-model trainer.
