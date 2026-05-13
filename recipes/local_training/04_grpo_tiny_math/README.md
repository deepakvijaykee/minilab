# 04 GRPO Tiny Math

Run verifier-reward RLVR on GSM8K-style math prompts.

```bash
bash recipes/local_training/04_grpo_tiny_math/run.sh
```

The default is deliberately small: `batch_size=1`, `num_generations=2`, and
`max_new_tokens=64`. Increase these only after the estimator and a short run
look reasonable on your GPU.
