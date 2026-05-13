# Expected Signals

- Training loss should trend downward over the run.
- Validation loss or perplexity should be finite when using a dataset with an
  eval split.
- Samples should move from random token sequences toward TinyStories-like text
  as `MAX_STEPS` increases.

The default run is intentionally short. Treat these as sanity checks until you
record measured runs for a specific GPU and preset.
