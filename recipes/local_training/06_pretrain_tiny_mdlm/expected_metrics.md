# Expected signals

The MDLM loss is a time-averaged denoising cross-entropy weighted by
the noise schedule. On a cosine schedule the high-noise timesteps,
where most tokens are masked, are upweighted relative to the easy
near-clean timesteps, so the absolute value of the loss is not
directly comparable with the autoregressive cross-entropy in recipe
01. What is comparable, and what to read instead, is the shape of the
curve. The first roughly 200 steps are a fast drop as the model fits
the marginal token distribution, after which the curve transitions
into a slower decline as the model learns context-conditioned
denoising. That slow phase is where the model is actually acquiring
the dependencies that produce coherent samples.

The checkpoint directory must contain `model.pt`, `config.json`,
`model_type.txt`, and `forward_process.json`. The last file records
the noise schedule, and the downstream diffusion recipes refuse to
load a checkpoint without it because they need that schedule to
reconstruct the forward process at training time. Treat it as part of
the model artifact rather than as metadata that can be regenerated.

The `--- Samples ---` block runs unconditional reverse sampling
starting from an all-mask sequence. MDLM supports unconditional
sampling because the schedule terminates at alpha equal to zero, which
means the all-mask state is on-distribution for the model's reverse
process. Block diffusion variants that lack a terminal mask prior
cannot sample unconditionally and print
`skipped: model requires clean x_0 context for reverse scoring`
instead, which is correct behavior for those families rather than a
sampling failure.

Sample quality at a thousand steps is poor, with token-shaped output
and broken syntax. This traces back to the same estimator property that
governs the loss curve. Each token is supervised through a noisy
timestep expectation rather than against a single direct cross-entropy
target, so the bias-variance balance of the per-token estimator sits in
a higher-variance regime than the autoregressive equivalent.
