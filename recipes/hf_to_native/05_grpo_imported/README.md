# 05 GRPO Imported Model

Run native Minilab RLVR on an imported-model alignment checkpoint.

```bash
bash recipes/hf_to_native/05_grpo_imported/run.sh
```

The default policy checkpoint is:

```text
checkpoints/imported/smollm2-135m-simpo/step_50
```

Point `POLICY_CHECKPOINT` at the SFT checkpoint if you want to skip preference
optimization.
