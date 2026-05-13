# 07 SFT Tiny Diffusion

Run response-only supervised fine-tuning for the pretrained diffusion LM.
Unlike AR SFT, this keeps the prompt tokens fixed and only noises/supervises
response tokens. That makes the recipe a diffusion-native instruction-tuning
path instead of treating the model as a next-token predictor.

```bash
bash recipes/local_training/07_sft_tiny_diffusion/run.sh
```

By default this consumes `checkpoints/local_training/diffusion/step_1000` and writes
to `checkpoints/local_training/diffusion_sft`.

Useful overrides:

```bash
MAX_STEPS=1000 MAX_EXAMPLES=5000 bash recipes/local_training/07_sft_tiny_diffusion/run.sh
DATASET=dolly bash recipes/local_training/07_sft_tiny_diffusion/run.sh
```
