# Expected signals

- Diffusion loss is a weighted denoising loss, not a per-token cross-entropy,
  so the absolute values do not match recipe 01. The shape is what matters:
  loss should drop quickly in the first ~200 steps then settle.
- The final checkpoint directory must contain `model.pt`, `config.json`,
  `model_type.txt`, and `forward_process.json`. Missing
  `forward_process.json` is the failure mode that breaks every downstream
  diffusion recipe.
- The `--- Samples ---` block runs unconditional reverse sampling. MDLM
  supports it. If you switch to a model that requires a clean x_0 context
  (some block-diffusion variants), the script prints `skipped: model
  requires clean x_0 context for reverse scoring` instead of samples.
- `run_metrics.json` is written under `checkpoints/local_training/diffusion/`.

Sample quality at 1000 steps is poor: tokens look text-like but sentences
do not. As with recipe 01, that is the small-scale ceiling.
