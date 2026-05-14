# Expected signals

- Diffusion DPO substitutes a one-sample ELBO of the denoising loss for
  the exact log-likelihood in standard DPO. Each preference pair therefore
  gives an unbiased but high-variance gradient estimate; per-pair noise
  can be on the order of the signal, which is why convergence looks
  rougher than AR DPO at matched step counts.
- VRPO targets that variance directly: it averages `--vrpo-num-samples`
  independent ELBO estimates per pair before computing the preference loss.
  Step time scales roughly linearly with that knob, and the variance of the
  preference-loss gradient drops as `1/num_samples` (so stderr like
  `1/sqrt(num_samples)`).
- The model line reads `(mdlm, <N> params, schedule=<schedule>)` for both
  algorithms. A `schedule=None` value means `forward_process.json` did not
  copy through; the trainer cannot reconstruct the forward process and the
  loss values are wrong from step 1.
- End-of-run generations come from reverse diffusion with
  `--sample-new-tokens 80`. Quality is bounded above by the diffusion SFT
  base. Preference tuning re-weights which response distribution the model
  denoises toward; it does not improve how well the model denoises.
