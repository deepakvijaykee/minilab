# 06 Pretrain tiny MDLM

Pretrains a masked diffusion LM with the `mdlm-25m` preset. This is the
diffusion-track counterpart to recipe 01; recipes 07-09 load this checkpoint.

```bash
bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
```

Defaults: `--preset mdlm-25m`, `--seq-len 512`, `--batch-size 4`,
`--max-steps 1000`, `--max-examples 10000`, gradient checkpointing on. The
batch size is smaller than recipe 01 (`8`) because MDLM keeps the full
sequence at every timestep and the activation budget is tighter.

Useful overrides:

```bash
MAX_STEPS=3000 MAX_EXAMPLES=50000 bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
DATASET=text8 SEQ_LEN=256 bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
```

The MDLM checkpoint directory carries an extra file alongside `model.pt`:
`forward_process.json` records the noise schedule. Every downstream
diffusion recipe loads that file too, so do not rename or drop it.
