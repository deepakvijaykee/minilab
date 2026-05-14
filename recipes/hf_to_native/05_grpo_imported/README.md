# 05 GRPO imported model

Native Minilab RLVR (GSM8K verifier reward) on an imported-and-aligned
checkpoint. By default it starts from the SimPO output of recipe 04; you
can skip the preference step by pointing `POLICY_CHECKPOINT` at the SFT
checkpoint from recipe 03 instead.

```bash
bash recipes/hf_to_native/05_grpo_imported/run.sh
```

Defaults: `MODEL=smollm2-135m`, `--algorithm grpo`,
`POLICY_CHECKPOINT=.../smollm2-135m-simpo/step_50`, `--max-steps 25`,
`--batch-size 1`, `--num-generations 2`, `--max-new-tokens 64`,
`--max-examples 100`, `--eval-examples 20`. Output:
`checkpoints/imported/smollm2-135m-grpo`.

The defaults are tiny because RLVR on a 135M model is much heavier per step
than on `gpt-10m`. Expect this run to take noticeably longer per step than
recipe 04. Bump `MAX_STEPS` and `NUM_GENERATIONS` together for any quality
signal; either one alone produces noise.
