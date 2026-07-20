# 03 Preference tiny

This recipe runs offline preference optimization on the SFT checkpoint
produced by recipe 02. The default algorithm is DPO on Anthropic
HH-RLHF, which is the cleanest entry point for reading what preference
tuning is doing at this scale: the trainer is increasing the
log-probability margin between chosen and rejected responses, with a
KL trust region around the SFT policy controlled by beta.

```bash
bash recipes/local_training/03_preference_tiny/run.sh
```

The defaults are `--algorithm dpo`, `--dataset hh`,
`SFT_CHECKPOINT=.../sft/step_500`, `--max-steps 300`, `--batch-size 2`,
`--lr 1e-5`, `--beta 0.1`, and `--max-examples 1000`. The save
directory is keyed by algorithm name, so DPO writes to
`checkpoints/local_training/preference_dpo` and the same convention
holds for every other variant.

DPO, IPO, and KTO need a frozen copy of the SFT model held in memory
as the reference, which roughly doubles the resident model weights. The
reference forward carries no gradients, so the added cost is weights, not
activations. SimPO, ORPO, CPO, and RePO
drop the explicit reference forward and replace it with a length- or
margin-based regularizer, which is meaningfully cheaper to run at this
scale.

```bash
ALGORITHM=simpo bash recipes/local_training/03_preference_tiny/run.sh
ALGORITHM=orpo bash recipes/local_training/03_preference_tiny/run.sh
```

The `--dataset` flag accepts `hh` and `ultrafeedback`.
