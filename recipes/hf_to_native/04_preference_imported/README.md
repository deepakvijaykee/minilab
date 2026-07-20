# 04 Preference imported model

This recipe runs native Minilab preference optimization on the SFT
checkpoint from recipe 03. The default algorithm is SimPO, which is
reference-free, and that choice is driven by memory rather than
preference for the algorithm. The imported base model is large enough
that holding a frozen reference copy alongside the trainable policy
is the difference between fitting on 8GB of VRAM and not, and SimPO
removes the second forward through a reference at the cost of using a
length-normalized margin in place of the reference KL.

```bash
bash recipes/hf_to_native/04_preference_imported/run.sh
```

The defaults are `MODEL=smollm2-135m`, `--algorithm simpo`,
`--dataset hh`, `SFT_CHECKPOINT=.../smollm2-135m-sft/step_100`,
`--max-steps 50`, `--batch-size 1`, `--lr 1e-5`, `--beta 0.1`, and
`--max-examples 200`. Output is written to
`checkpoints/imported/smollm2-135m-simpo`.

Switch to DPO when you want the explicit frozen-reference comparison
and have the memory to hold a second copy of the model:

```bash
ALGORITHM=dpo bash recipes/hf_to_native/04_preference_imported/run.sh
```

The reference checkpoint path is resolved by
`resolve_reference_path` in `minilab.alignment`. For the imported
pipeline it defaults to the same SFT checkpoint that was used as the
trainable starting point, which is the standard choice in the DPO
literature: the reference is the policy immediately before
preference tuning, so the KL is measuring movement away from that
exact distribution.
