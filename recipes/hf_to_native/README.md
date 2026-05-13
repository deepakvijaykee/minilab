# Hugging Face to Native Recipes

This track is intentionally narrow: curated sub-1B Hugging Face causal LMs for
inspection, generation, and import into Minilab's native checkpoint format.

Install the optional dependencies:

```bash
python -m pip install -e ".[data,hf]"
```

If `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, or `TRANSFORMERS_CACHE` is not set, the
HF scripts default to `.cache/huggingface/` under the repo so downloads stay
inside the workspace.

## Curated Models

| Alias | Hugging Face repo | Size | Role |
| --- | --- | ---: | --- |
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M` | 135M | smallest pretrained baseline |
| `smollm2-135m-instruct` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | tiny instruct baseline |
| `gemma3-270m` | `google/gemma-3-270m` | 270M | small modern pretrained baseline |
| `gemma3-270m-it` | `google/gemma-3-270m-it` | 270M | small modern instruct baseline |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M` | 360M | lightweight pretrained baseline |
| `smollm2-360m-instruct` | `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | lightweight instruct baseline |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 0.6B | main modern sub-1B baseline |
| `qwen3-0.6b-base` | `Qwen/Qwen3-0.6B-Base` | 0.6B | main modern sub-1B pretrained baseline |

## Commands

Inspect a preset without loading weights:

```bash
python scripts/hf_inspect.py --model smollm2-135m
```

Load weights and measure parameter memory:

```bash
python scripts/hf_inspect.py --model smollm2-135m --load
```

Generate text:

```bash
python scripts/hf_generate.py \
  --model smollm2-135m-instruct \
  --prompt "Explain gravity in one paragraph." \
  --device cuda
```

Import a compatible HF model to native Minilab format:

```bash
bash recipes/hf_to_native/02_import/run.sh
```

Then run native training recipes on the imported checkpoint:

```bash
bash recipes/hf_to_native/03_sft_imported/run.sh
bash recipes/hf_to_native/04_preference_imported/run.sh
bash recipes/hf_to_native/05_grpo_imported/run.sh
bash recipes/hf_to_native/06_summary/run.sh
```

List curated presets:

```bash
python scripts/hf_inspect.py --list-presets
```

## Scope

These scripts do not replace Minilab's native model/training stack. HF import is
for bringing real pretrained small-model baselines into the same trainers used
by local pretraining, SFT, preference optimization, and RLVR.
