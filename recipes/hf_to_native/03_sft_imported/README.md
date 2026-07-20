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

Native Q/V LoRA can be enabled directly on any compatible imported checkpoint:

```bash
python scripts/sft.py \
  --tokenizer checkpoints/imported/qwen3-0.6b/tokenizer.json \
  --checkpoint checkpoints/imported/qwen3-0.6b \
  --save-dir checkpoints/imported/qwen3-0.6b-lora-sft \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-steps 100
```

Only adapter parameters are optimizer-owned. Resume restores the saved adapter
structure, so `--lora-rank` and `--lora-alpha` are checkpoint-creation options
and must not be repeated with `--resume-from`.

Imported instruction tokenizers serialize the complete supervised conversation
with their saved chat template. The prompt prefix is masked from the loss and
the assistant response, including its turn terminator, is labeled. Use
`--dataset structured_output` for the deterministic exact-envelope curriculum;
unlike open-ended Alpaca loading, that curriculum rejects any example that
would require sequence truncation.
