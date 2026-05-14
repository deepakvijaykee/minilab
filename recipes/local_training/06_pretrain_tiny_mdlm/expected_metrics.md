# Expected signals

- MDLM's loss is a time-averaged denoising cross-entropy weighted by the
  schedule. On a cosine schedule the high-noise (mostly masked)
  timesteps get up-weighted relative to easy near-clean ones, so the
  absolute loss is not comparable to recipe 01's AR cross-entropy. What
  to read is the shape: a fast drop in the first ~200 steps as the model
  learns the marginal token distribution, then a slower decline as it
  learns context-conditioned denoising.
- The checkpoint directory must contain `model.pt`, `config.json`,
  `model_type.txt`, and `forward_process.json`. The last file records the
  noise schedule; without it the downstream recipes cannot reconstruct
  the forward process and refuse to load.
- `--- Samples ---` runs unconditional reverse sampling from all `[MASK]`.
  MDLM supports this because the schedule has `alpha[-1]=0`. Block
  diffusion variants without a terminal mask prior cannot sample
  unconditionally and print `skipped: model requires clean x_0 context
  for reverse scoring` instead.

Sample quality at 1000 steps is poor: text-shaped tokens, broken syntax.
Diffusion LMs at this scale need more compute than AR for matching
qualitative output because each token is supervised through a noisy
expectation over timesteps rather than a single direct cross-entropy.
