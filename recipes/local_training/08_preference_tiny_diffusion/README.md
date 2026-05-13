# 08 Preference Tiny Diffusion

Run diffusion preference optimization on the diffusion SFT checkpoint.
The default is diffusion DPO, which compares chosen and rejected responses
through the model's diffusion loss proxy. `ALGORITHM=vrpo` runs the
variance-reduced variant that averages multiple shared diffusion estimates per
pair.

```bash
bash recipes/local_training/08_preference_tiny_diffusion/run.sh
```

Default algorithm: `dpo`.

You can also run the variance-reduced objective:

```bash
ALGORITHM=vrpo bash recipes/local_training/08_preference_tiny_diffusion/run.sh
```

Useful overrides:

```bash
DATASET=ultrafeedback bash recipes/local_training/08_preference_tiny_diffusion/run.sh
MAX_STEPS=1000 MAX_EXAMPLES=5000 bash recipes/local_training/08_preference_tiny_diffusion/run.sh
```
