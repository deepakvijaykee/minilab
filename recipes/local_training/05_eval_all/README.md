# 05 Eval all

Runs `scripts/evaluate.py` over the four AR checkpoints produced by recipes
01 through 04. Missing checkpoints are skipped rather than failing the
whole pass, so this is safe to run after any partial path.

```bash
bash recipes/local_training/05_eval_all/run.sh
```

The recipe iterates over `base`, `sft`, `preference`, and `grpo` labels and
calls `evaluate.py` for each, which prints validation perplexity plus
generation diversity metrics (Distinct-1/2/3 and Self-BLEU-4) and a few
samples. Diversity moves the most across the path: SFT and preference
checkpoints usually drop Distinct-N relative to the base, and good RLVR
runs claw it partway back.

Use this to spot regressions, not to publish numbers; the eval is sampled,
deterministic in slice but small.
