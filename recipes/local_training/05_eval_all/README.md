# 05 Eval all

This recipe runs `scripts/evaluate.py` over the four autoregressive
checkpoints produced by recipes 01 through 04. Missing checkpoints are
skipped with a notice rather than failing the pass, so the recipe is
safe to invoke after any partial run, which is useful for inspecting
where the pipeline stands without restarting from the beginning.

```bash
bash recipes/local_training/05_eval_all/run.sh
```

The recipe iterates over the labels `base`, `sft`, `preference`, and
`grpo`, calling `evaluate.py` for each one in turn. Each invocation
prints validation perplexity, a small block of generation diversity
metrics (Distinct-1, Distinct-2, Distinct-3, and Self-BLEU-4), and a
few sampled completions. The most informative axis at this scale is
not perplexity but diversity. SFT and preference tuning both shrink
the output distribution, which shows up as a Distinct-N drop relative
to the base, while a well-behaved RLVR run partially restores
diversity because the verifier reward does not collapse onto a single
canonical answer the way preference labels can.

This eval is for spotting regressions and reading the shape of the
distributional shift across stages. It is not sized to publish
numbers: the sample slice is deterministic but small, so a difference
of one or two percentage points between stages is well inside the
sampling noise floor.
