# 04 Preference imported model

Native Minilab preference optimization on the SFT checkpoint from recipe
03. Default is SimPO, which is reference-free; that matters here because
the imported base model is large enough that keeping a frozen reference
copy in memory is the difference between fitting on the GPU and not.

```bash
bash recipes/hf_to_native/04_preference_imported/run.sh
```

Defaults: `MODEL=smollm2-135m`, `--algorithm simpo`, `--dataset hh`,
`SFT_CHECKPOINT=.../smollm2-135m-sft/step_100`, `--max-steps 50`,
`--batch-size 1`, `--lr 1e-5`, `--beta 0.1`, `--max-examples 200`. Output:
`checkpoints/imported/smollm2-135m-simpo`.

Use DPO when you want the frozen-reference comparison and have the memory
for it:

```bash
ALGORITHM=dpo bash recipes/hf_to_native/04_preference_imported/run.sh
```

The reference path is resolved by `resolve_reference_path` in
`minilab.alignment`; for the imported pipeline it defaults to the same SFT
checkpoint that was used as the trainable starting point.
