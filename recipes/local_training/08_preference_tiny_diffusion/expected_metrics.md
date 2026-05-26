# Expected signals

Diffusion DPO substitutes a one-sample ELBO of the denoising loss for
the exact log-likelihood that standard DPO uses. The estimator is
unbiased but high-variance, and each preference pair therefore
contributes a noisy gradient. Per-pair noise can be on the same order
as the signal, which is the structural reason convergence looks
visibly rougher than autoregressive DPO at matched step counts. The
right way to read the loss curve is as a noisy random walk around a
slowly descending trend, not as a smooth descent.

VRPO targets that variance directly. It averages
`--vrpo-num-samples` independent ELBO estimates per pair before
computing the preference loss, which means the variance of the
preference-loss gradient drops as one over the number of samples and
the standard error therefore drops as one over the square root of the
number of samples. Step time grows roughly linearly in that count.
The intended trade-off, more compute per step in exchange for a much
smoother optimization trajectory, is the one a practitioner usually
wants on a tiny budget where each gradient step is precious.

The model line at startup reads
`(mdlm, <N> params, schedule=<schedule>)` for both algorithms. A
`schedule=None` value is a load failure in disguise:
`forward_process.json` did not copy through, the trainer cannot
reconstruct the forward process, and the loss values are wrong from
step 1 even though no exception was raised. The fix is to recopy or
regenerate that file from the SFT checkpoint directory.

The end-of-run generations come from reverse diffusion with
`--sample-new-tokens 80`. Their quality is bounded above by the
diffusion SFT base. Preference tuning shifts which response
distribution the model denoises toward but does not improve how well
the model denoises in absolute terms, which means the qualitative
ceiling of recipes 06 and 07 also caps the qualitative ceiling here.
