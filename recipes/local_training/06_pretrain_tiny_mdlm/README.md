# 06 Pretrain tiny MDLM

Pretrains a masked diffusion LM with the `mdlm-25m` preset. This is the
diffusion-track counterpart to recipe 01; recipes 07-09 load this checkpoint.

```bash
bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
```

It pretrains `mdlm-25m` for 1000 steps at batch 4 over 10000 TinyStories
examples, with gradient checkpointing on. The batch size is half of
recipe 01 (`8`) because MDLM keeps the full sequence at every timestep,
which is tighter on activation memory.

Useful overrides:

```bash
MAX_STEPS=3000 MAX_EXAMPLES=50000 bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
DATASET=text8 SEQ_LEN=256 bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
```

The MDLM checkpoint directory carries an extra file alongside `model.pt`:
`forward_process.json` records the noise schedule. Every downstream
diffusion recipe loads that file too, so do not rename or drop it.

At matched parameters, diffusion pretraining needs more samples than AR
because each token is supervised through a stochastic timestep rather
than a deterministic next-token target. Plan on roughly an order of
magnitude more steps than recipe 01 for comparable coherence.
