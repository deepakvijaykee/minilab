# Hugging Face to native recipes

These recipes bring curated sub-1B Hugging Face causal language models
through the same Minilab trainers as the local-training recipes, so the
alignment behavior of a real pretrained baseline can be observed under
the same code path as the tiny from-scratch experiments. This is not a
general Hugging Face loader: its one purpose is the comparison between
local pretrains and curated baselines under identical alignment code.

Install the optional dependencies:

```bash
python -m pip install -e ".[data,hf]"
```

If none of `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, or `TRANSFORMERS_CACHE`
is set, the HF scripts cache into `.cache/huggingface/` under the repo
so downloads stay inside the workspace. That default exists mostly so
the experiments are reproducible without polluting a shared system
cache.

## Curated models

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

Import a compatible HF model to the native Minilab format:

```bash
bash recipes/hf_to_native/02_import/run.sh
```

Then run native training recipes on the imported checkpoint:

```bash
bash recipes/hf_to_native/03_sft_imported/run.sh
bash recipes/hf_to_native/04_preference_imported/run.sh
bash recipes/hf_to_native/05_grpo_imported/run.sh
```

List curated presets:

```bash
python scripts/hf_inspect.py --list-presets
```

## Scope

The import step maps Hugging Face weights into Minilab's native GPT
format so that the resulting checkpoint loads through the same code
path as a from-scratch checkpoint. After import, every native trainer,
sampler, and evaluator works without modification. Llama-compatible SmolLM2
and dense Qwen3 models import natively today. Qwen3 mapping includes its
explicit attention head dimension and per-head Q/K normalization and is
guarded by logit verification. Gemma3 remains inspection/generation-only until
its native weight mapping is validated. The model-type guard is
`_native_config` in `scripts/import_hf.py`, which is where a new family
is added once its mapping is verified.

Qwen3 requires Transformers 4.51 or newer. Import and verify it with:

```bash
MODEL=qwen3-0.6b DEVICE=cpu bash recipes/hf_to_native/02_import/run.sh
```

Native alignment on an imported instruction model depends on one fact: the
tokenizer carries its upstream chat template, and honoring that template
keeps supervision aligned with the model's own turn structure. At both the
generation and the supervised-training boundary the imported tokenizer
applies that template, so SFT and preference losses fall only on the
assistant response tokens, including the upstream turn terminator. Everything outside that boundary stays plain text:
ordinary `encode()` calls and completion decoding are unaffected.
Exact-output Qwen3 runs use non-thinking mode, so the model does not emit
a reasoning prefix that would break a strict code or tool envelope.
Agentic and supervised examples rebuild the full user -> assistant ->
environment -> assistant context, while the native tokenizer keeps its
original token-concatenation path underneath.

When a tokenizer is passed to ordinary generation, its saved EOS token
terminates the response as well. Raw-code tasks need this: without an
application-level closing tag, the decoder would otherwise run past a
correct function straight into a second turn.

Use `scripts/sft.py --dataset structured_output` to train the exact
raw-Python, tool-call, and final-answer contracts. That curriculum drops
any example that would be truncated rather than teach a broken envelope.
The gate before group-relative RL is a group with mixed rewards: until
the verifier sees both successes and failures on the same prompt, the
group-relative advantage is zero and RL has no gradient to follow.
