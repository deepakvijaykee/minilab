# 06 Pretrain Tiny MDLM

Pretrain a small masked diffusion language model with the `mdlm-25m` preset.
This creates the base checkpoint used by the diffusion SFT, preference, and
GRPO recipes.

```bash
bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
```

This is the diffusion counterpart to `01_pretrain_tiny_gpt`. It trains an MDLM
checkpoint that can be used by the diffusion SFT, preference, and GRPO recipes.

Useful overrides:

```bash
MAX_STEPS=3000 MAX_EXAMPLES=50000 bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
DATASET=text8 SEQ_LEN=256 bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
```
