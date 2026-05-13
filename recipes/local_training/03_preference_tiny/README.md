# 03 Preference Tiny

Run offline preference optimization on the SFT checkpoint.

```bash
bash recipes/local_training/03_preference_tiny/run.sh
```

Default algorithm: `dpo`.

Reference-free alternatives are useful on tighter memory budgets:

```bash
ALGORITHM=simpo bash recipes/local_training/03_preference_tiny/run.sh
ALGORITHM=orpo bash recipes/local_training/03_preference_tiny/run.sh
```
