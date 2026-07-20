# 02 Import HF to native

This recipe imports a compatible Hugging Face causal language model
into Minilab's native GPT checkpoint format. After the import the
model loads through exactly the same code path as a checkpoint trained
from scratch in the local-training track, so the later alignment
recipes can pick it up without any HF-specific glue.

```bash
bash recipes/hf_to_native/02_import/run.sh
```

Out of the box the recipe pulls `smollm2-135m`, writes the converted
checkpoint to `checkpoints/imported/smollm2-135m`, sets the native
context length to 512, runs on CPU, and verifies logits. The verify
step is where the import is actually checked: it forwards both the original
HF model and the converted native model on a short prompt and reports
the maximum and mean absolute logit difference. Values around 1e-5 or
smaller mean the mapping is clean and any remaining difference is
floating-point accumulation order rather than a structural bug.

After import, the standard Minilab training scripts work directly on
the imported checkpoint:

```bash
python scripts/sft.py \
  --tokenizer checkpoints/imported/smollm2-135m/tokenizer.json \
  --checkpoint checkpoints/imported/smollm2-135m \
  --save-dir checkpoints/imported/smollm2-135m-sft
```

The importer accepts Llama-compatible SmolLM2 and dense Qwen3 models. Qwen3 is
not treated as a Llama alias: the mapping preserves its explicit attention head
dimension and Q/K normalization parameters, and `--verify` compares native and
source logits. Gemma3 still trips the model-type guard. That guard matters because the
failure it prevents is silent: a model whose conventions differ from what
the mapping assumes, its attention-bias or embedding-tying handling for
instance, would otherwise import into a checkpoint that loads cleanly and
only then generates wrong logits.

```bash
MODEL=qwen3-0.6b DEVICE=cpu bash recipes/hf_to_native/02_import/run.sh
```
