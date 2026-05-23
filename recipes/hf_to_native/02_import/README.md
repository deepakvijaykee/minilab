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
step is the load-bearing safety check: it forwards both the original
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

The importer currently accepts only Llama-compatible Hugging Face
models, which today means SmolLM2. Qwen3 and Gemma3 trip the
model-type guard in `scripts/import_hf.py` and require separate
weight-mapping work before they can come through this path. The guard
is intentional: silently importing a model whose attention bias or
embedding tying differs from Llama would produce a checkpoint that
loads but generates incorrect logits, which is a harder failure to
debug than a refused import.
