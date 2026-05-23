# Expected signals

The training loss is recipe 06's denoising objective restricted to the
response tokens. Prompt tokens stay clean at every timestep and only
the response is noised and supervised. Step-to-step jitter is visibly
larger than the autoregressive SFT loss because each step samples a
single timestep per example and the loss weight depends on that
sampled timestep. This is the diffusion-side analogue of how AR SFT's
loss is noise-free; the noise has not vanished, it has moved into the
estimator.

The checkpoint directory contains `model.pt`, `config.json`,
`model_type.txt`, and `forward_process.json`. Recipes 08 and 09 refuse
to load this checkpoint without the forward-process file, because the
forward process is what defines the diffusion loss they need to
compute. As in recipe 06, that file is part of the model rather than
auxiliary data.

The `--- After Diffusion SFT ---` block generates by reverse diffusion
with the prompt held clean. This is structurally infilling rather than
left-to-right sampling, which is why the diffusion track is a
genuinely different alignment story rather than a relabeled
autoregressive one. The qualitative ceiling on this output is set by
recipe 06: SFT can shift which response distribution the model
denoises toward, but it cannot teach the model to denoise text-shaped
outputs when the base does not already produce them. With the default
thousand-step base, the right expectation is short, choppy answers
that show the response template forming around an undertrained
density model.
