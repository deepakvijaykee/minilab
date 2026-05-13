# 02 SFT Tiny Instruct

Supervised fine-tune the pretrained tiny LM on Alpaca-style instruction data.

```bash
bash recipes/local_training/02_sft_tiny_instruct/run.sh
```

By default this consumes `checkpoints/local_training/lm/step_1000` and writes to
`checkpoints/local_training/sft`.
