# 03 Preference tiny

Offline preference optimization on the SFT checkpoint. Default algorithm is
DPO on Anthropic HH-RLHF.

```bash
bash recipes/local_training/03_preference_tiny/run.sh
```

Defaults: `--algorithm dpo`, `--dataset hh`, `SFT_CHECKPOINT=.../sft/step_500`,
`--max-steps 300`, `--batch-size 2`, `--lr 1e-5`, `--beta 0.1`,
`--max-examples 1000`. The save directory is named after the algorithm:
`checkpoints/local_training/preference_<algorithm>`.

DPO, IPO, and KTO keep a frozen copy of the SFT model as the reference and
need roughly 2x the activation memory. SimPO, ORPO, CPO, and RePO drop the
reference and are noticeably cheaper:

```bash
ALGORITHM=simpo bash recipes/local_training/03_preference_tiny/run.sh
ALGORITHM=orpo bash recipes/local_training/03_preference_tiny/run.sh
```

The `--dataset` flag accepts `hh` and `ultrafeedback`. Note the diffusion
counterpart (recipe 08) uses the string `hh-rlhf` instead of `hh` because
the diffusion preference loaders are wired up with different names; that
mismatch is real and lives in the loader registries inside each script.
