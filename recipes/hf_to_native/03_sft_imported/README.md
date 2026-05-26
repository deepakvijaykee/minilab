# 03 SFT imported model

This recipe runs native Minilab SFT on a checkpoint imported with
recipe 02. It uses the same `scripts/sft.py` as the local-training
track, with only the input checkpoint and the learning rate
differing. Running SFT through the native code path rather than
through the Hugging Face trainer is what lets the imported baseline
be compared directly with a from-scratch checkpoint under matched
alignment code, which is the motivation for the import path in the
first place.

```bash
bash recipes/hf_to_native/03_sft_imported/run.sh
```

The defaults are `MODEL=smollm2-135m`,
`CHECKPOINT=checkpoints/imported/smollm2-135m`, `--max-steps 100`,
`--batch-size 1`, `--lr 2e-5`, and `--max-examples 500`. Output lands
in `checkpoints/imported/smollm2-135m-sft`.

The learning rate, 2e-5, is much lower than the 1e-4 used by the
from-scratch SFT recipe. The reason is that the imported model
already has competent representations, and a rate set for a model
starting from a partially-trained TinyStories base would damage those
representations rather than refine them. SFT on a pretrained baseline
is structurally a fine-tuning problem, and the conservative learning
rate reflects that.

```bash
MODEL=smollm2-360m MAX_STEPS=200 bash recipes/hf_to_native/03_sft_imported/run.sh
```
