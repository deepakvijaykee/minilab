# 06 Pretrain tiny MDLM

This recipe pretrains a masked diffusion language model under the
`mdlm-25m` preset. It is the diffusion-track counterpart to recipe 01,
and recipes 07 through 09 load the checkpoint it produces. The
diffusion branch is set up so that the model is never coerced into a
next-token predictor for alignment. Every downstream stage keeps the
diffusion objective, so the comparison with the autoregressive branch
holds all the way through alignment rather than only at pretraining.

```bash
bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
```

The defaults are a thousand training steps at `batch_size=4` over ten
thousand TinyStories examples, with gradient checkpointing enabled.
The batch size is half of recipe 01's eight because MDLM keeps the
full sequence resident at every timestep rather than only the prefix
up to the current position, which tightens the activation-memory
budget for the same sequence length.

Useful overrides:

```bash
MAX_STEPS=3000 MAX_EXAMPLES=50000 bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
DATASET=text8 SEQ_LEN=256 bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
```

The MDLM checkpoint directory carries one extra file beyond what the
autoregressive recipes write: `forward_process.json`, which records
the noise schedule. Every downstream diffusion recipe needs this file
to reconstruct the forward process at training time and refuses to
load a checkpoint without it. Renaming or dropping it is the most
common reason later recipes fail to start, so treat it as part of the
model, not as auxiliary metadata you can regenerate.

At matched parameters and matched compute, diffusion pretraining needs
substantially more samples than autoregressive pretraining to reach
the same coherence. Each token in the diffusion loss is supervised through a
stochastic timestep expectation rather than against a deterministic
next-token target, so the same gradient budget carries strictly less
information per token. Plan for roughly an order of magnitude more steps
than recipe 01 when you are chasing comparable sample quality.
