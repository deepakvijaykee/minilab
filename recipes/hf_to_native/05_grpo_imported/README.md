# 05 GRPO imported model

Native Minilab RLVR (GSM8K verifier reward) on an imported-and-aligned
checkpoint. By default it starts from the SimPO output of recipe 04; you
can skip the preference step by pointing `POLICY_CHECKPOINT` at the SFT
checkpoint from recipe 03 instead.

```bash
bash recipes/hf_to_native/05_grpo_imported/run.sh
```

Default config: `smollm2-135m` SimPO policy, GRPO for 25 outer steps at
batch 1 with 2 generations per prompt, 64 max new tokens, 100 training
examples, 20 eval examples. Writes to
`checkpoints/imported/smollm2-135m-grpo`.

The defaults are tiny because RLVR on a 135M model is much heavier per step
than on `gpt-10m`. Expect this run to take noticeably longer per step than
recipe 04. Bump `MAX_STEPS` and `NUM_GENERATIONS` together for any quality
signal; either one alone produces noise.
