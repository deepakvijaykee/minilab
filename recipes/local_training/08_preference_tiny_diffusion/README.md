# 08 Preference tiny diffusion

This recipe runs diffusion preference optimization on the diffusion
SFT checkpoint produced by recipe 07. The default algorithm is
diffusion DPO, which substitutes a one-sample ELBO of the denoising
loss for the exact log-likelihood that standard DPO uses on
autoregressive models. The variance-reduced variant, VRPO, averages
multiple independent denoising estimates per pair before computing
the preference loss, and is the recommended choice when the
high-variance one-sample estimator makes convergence look unstable.

```bash
bash recipes/local_training/08_preference_tiny_diffusion/run.sh
```

The default run is diffusion DPO on HH-RLHF starting from
`diffusion_sft/step_500`, with 300 steps at `batch_size=2`,
`lr=1e-5`, `beta=0.1`, 1000 preference pairs, and 80 sampled response
tokens. The save directory is named after the algorithm:
`checkpoints/local_training/diffusion_<algorithm>`.

Run the variance-reduced variant with:

```bash
ALGORITHM=vrpo bash recipes/local_training/08_preference_tiny_diffusion/run.sh
```

VRPO adds `--vrpo-num-samples`, which defaults to four and is the
knob that trades compute for variance. Step time grows roughly
linearly in that count because each pair forwards through the
diffusion loss that many times, and the estimator switches its method
to `diffusion_vrpo` automatically so the VRAM report matches the
actual call shape.

The diffusion preference loaders accept the dataset names `hh-rlhf`
and `ultrafeedback`. The autoregressive counterpart in recipe 03 uses
`hh` rather than `hh-rlhf`. Both ultimately resolve to the same
HH-RLHF data loader, but they are registered under different names in
their respective scripts, so the spelling has to match the script in
use rather than the dataset itself.
