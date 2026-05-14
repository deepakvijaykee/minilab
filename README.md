# minilab

Minilab is my small-scale language-model training lab.

The goal is to make the full pretraining → SFT → preference optimization →
RLVR loop runnable on a single consumer GPU, ideally an 8GB laptop GPU.

This is not a distributed training framework. It is not trying to replace
TRL, torchtune, Axolotl, or Megatron. It is for understanding and debugging
post-training methods in small, inspectable experiments where I can see
what each step does to the model.

The main path is:

```text
tokenizer → tiny GPT pretraining → SFT → DPO/SimPO → GRPO with GSM8K verifier → eval
```

Each step is a recipe under `recipes/local_training/`, runnable as one
`bash run.sh`. The diffusion branch (recipes `06`-`09`) mirrors the same
shape with MDLM in place of GPT.

## Cost on my laptop

Hardware: NVIDIA RTX 5060 Laptop, 8GB VRAM.
Main tested path: `gpt-10m`, 4k-vocab BPE tokenizer, FP32 + AdamW, grad
checkpointing where the recipe enables it.

Wall times include the first `torch.compile` cold-start (~30-60s) and the
first-run HF dataset download; both cache after that. Peak VRAM is also
recorded in `run_metrics.json` as `max_memory_reserved_gb`.

| Stage | Steps | Peak VRAM | Wall time | Result |
| --- | ---: | ---: | ---: | --- |
| 00 tokenizer | — | CPU | ~30s | tokenizer saved, sample sentence roundtrips |
| 01 pretrain `gpt-10m` | 1000 | ~1.5 GB | ~3 min | loss curve looks right; samples have TinyStories cadence but aren't stories yet |
| 02 SFT (Alpaca) | 500 | ~1.2 GB | ~2 min | output shifts from story drift to Q/A shape (content still weak) |
| 03 DPO (HH-RLHF) | 300 | ~1.3 GB | ~2 min | chosen margin stays positive on most pairs |
| 04 GRPO (GSM8K) | 100 | ~1.2 GB | ~12 min | the rollout loop is the wall-time cost; GSM8K accuracy is single-digit and noisy |
| 05 eval | — | ~0.8 GB | ~3 min | per-stage perplexity, Distinct-N, and five sampled completions |
| 06 pretrain `mdlm-25m` | 1000 | ~2.0 GB | ~5 min | denoising loss trends down; samples are token-shaped but not coherent |
| 07 diffusion SFT | 500 | ~1.7 GB | ~3 min | response-token loss drops; ceiling is whatever recipe 06 produced |
| 08 diffusion DPO | 300 | ~2.5 GB | ~6 min | trainable plus frozen reference both fit on 8GB; preference loss stays finite |
| 09 diffusion GRPO | 100 | ~2.3 GB | ~30 min | 64 reverse-diffusion forwards × 2 generations × 100 outer steps is where the time goes |

Peak VRAM never crosses 3 GB at the defaults, which leaves room to push
`PRESET=gpt-25m`, a larger `BATCH_SIZE`, or longer `MAX_NEW_TOKENS` before
the 8GB limit bites. Run `scripts/estimate_vram.py` before pushing any
of those up.

## Known limitations

- The default runs are sanity checks, not leaderboard runs. Coherent
  TinyStories prose and useful instruction quality need 3000+ pretrain
  steps and a larger preset (`gpt-25m` or `gpt-60m`).
- Tiny models pick up formatting before reasoning, so SFT and DPO/SimPO
  improvements show up in response shape long before they show up in
  GSM8K-style accuracy.
- GRPO/RLVR can stay at 0% indefinitely if the SFT base never produces
  answer-shaped completions. The verifier has nothing to credit, so the
  policy gradient is zero. Train recipe 02 longer before chasing RLVR.
- Diffusion alignment is the less-validated branch. The default recipes
  run to completion; qualitative output at the laptop scale is still poor.
- The main tested path is GPT-style tiny models. Mamba, Hymba, xLSTM,
  ByteLatent, and the diffusion variants are in the registry so the
  comparison scripts have something to compare against, not because they
  have been driven end to end through alignment.
- HF import is Llama-only today. Qwen3 and Gemma3 round-trip through
  inspection and generation but are rejected by `scripts/import_hf.py`
  until their weight mappings are validated.

## Install

Requires Python 3.10 or newer.

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[data]"     # dataset-backed scripts
python -m pip install -e ".[logging]"  # aim logging
python -m pip install -e ".[dev]"      # pytest and ruff
```

## Local training recipes

The main path lives in `recipes/local_training/`. Install the data extra
first because the recipes load TinyStories, Alpaca, HH-RLHF or
UltraFeedback, and GSM8K through Hugging Face Datasets:

```bash
python -m pip install -e ".[data]"
```

Autoregressive run order:

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
bash recipes/local_training/02_sft_tiny_instruct/run.sh
bash recipes/local_training/03_preference_tiny/run.sh
bash recipes/local_training/04_grpo_tiny_math/run.sh
bash recipes/local_training/05_eval_all/run.sh
```

Diffusion run order (uses the same tokenizer):

```bash
bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
bash recipes/local_training/07_sft_tiny_diffusion/run.sh
bash recipes/local_training/08_preference_tiny_diffusion/run.sh
bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

The diffusion branch is parallel to the AR branch stage for stage, without
treating diffusion models as next-token predictors:

| Stage | AR path | Diffusion path |
| --- | --- | --- |
| Pretraining | next-token prediction | masked denoising |
| SFT | response-only label loss | prompt-fixed response denoising |
| Preference tuning | DPO/IPO/SimPO/etc. over response log-probs | DPO/VRPO over diffusion loss proxies |
| RLVR | GRPO/RLOO/etc. with generated completions | GRPO over reverse denoising trajectories |

Every recipe carries a `README.md`, `config.yaml`, `run.sh`,
`expected_metrics.md`, and `sample_outputs.md`. Scale up with environment
overrides like `PRESET=gpt-25m`, `MAX_STEPS=3000`, or `ALGORITHM=simpo`.

## Tiny presets

Presets are pre-sized model configs that fit common laptop budgets:

| Preset | Family | Default context | Approx params | Use case |
| --- | --- | ---: | ---: | --- |
| `gpt-10m` | GPT | 512 | ~7M-10M | default tiny training path |
| `gpt-25m` | GPT | 512 | ~20M-25M | larger SFT/preference runs |
| `gpt-60m` | GPT | 1024 | ~52M-59M | stretch local GPT runs |
| `mamba-25m` | Mamba | 512 | ~22M-29M | SSM comparison runs |
| `mdlm-25m` | MDLM | 512 | ~26M-31M | diffusion LM experiments |

Parameter counts vary with tokenizer vocabulary size. The ranges above
cover the recipe default 4k vocabulary through a 16k vocabulary.

Use presets directly:

```bash
python scripts/pretrain_lm.py --tokenizer tokenizer.json --preset gpt-10m
python scripts/sft.py --tokenizer tokenizer.json --preset gpt-10m
python scripts/pretrain_diffusion.py --tokenizer tokenizer.json --preset mdlm-25m
```

Estimate memory before a run:

```bash
python scripts/estimate_vram.py \
  --model gpt-25m \
  --method grpo \
  --seq-len 512 \
  --batch-size 1 \
  --num-generations 4
```

Training runs automatically write PyTorch allocator measurements to
`run_metrics.json` in the final checkpoint directory and in the recipe save
directory. On CUDA, this includes `max_memory_allocated_gb` and
`max_memory_reserved_gb` from `torch.cuda` peak memory stats.

## Hugging Face bridge

The HF bridge is for curated sub-1B causal LMs: inspect them, generate
from them, or import compatible checkpoints into the native Minilab
format so they go through the same trainers as everything else. This is
not a general HF loader. Only Llama-compatible weights are wired up today
(SmolLM2 works; Qwen3 and Gemma3 round-trip through inspection and
generation but are rejected by import until their weight mappings are
validated).

```bash
python -m pip install -e ".[data,hf]"
python scripts/hf_inspect.py --list-presets
python scripts/hf_generate.py --model smollm2-135m-instruct --device cuda
bash recipes/hf_to_native/02_import/run.sh
bash recipes/hf_to_native/03_sft_imported/run.sh
bash recipes/hf_to_native/04_preference_imported/run.sh
bash recipes/hf_to_native/05_grpo_imported/run.sh
```

Curated aliases include `smollm2-135m`, `smollm2-360m`, `gemma3-270m`, and
`qwen3-0.6b`, plus instruct/base variants where available. The full list
comes from `scripts/hf_inspect.py --list-presets`. Recipes live in
`recipes/hf_to_native/`.

## Running scripts directly

If you would rather skip the recipe wrappers, the underlying scripts take
the same flags:

```bash
python scripts/train_tokenizer.py --save tokenizer.json
python scripts/pretrain_lm.py --tokenizer tokenizer.json
python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
```

```bash
python scripts/sft.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
python scripts/preference.py --algorithm dpo --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
```

```bash
python scripts/pretrain_diffusion.py --tokenizer tokenizer.json --model mdlm
python scripts/sample_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
```

```bash
python scripts/sft_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
python scripts/dpo_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion_sft/step_3000
python scripts/grpo_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion_sft/step_3000
```

## What else is in here

Beyond the main path, the registry includes alternative implementations
that are present mostly so the comparison scripts have something to
compare against:

- Other LM families: Mamba/Mamba-2, Hybrid, Hymba, xLSTM, ByteLatent.
- Diffusion LMs: MDLM, SEDD, D3PM, block diffusion.
- Preference-optimization variants: IPO, CPO, ORPO, RePO, KTO alongside
  DPO and SimPO.
- Online RL variants: RLOO, GSPO, DAPO, PPO alongside GRPO.
- Tokenizer variants: BPE, WordPiece, Unigram, character, byte.

The comparison scripts (`scripts/compare_attention.py`,
`scripts/compare_position.py`, `scripts/compare_connection.py`,
`scripts/compare_diffusion.py`) are the closest thing to first-class entry
points for these. The full alignment pipeline has only been driven end to
end on GPT-style models.

## Package contents

The package is registry-based: models, tokenizers, attention layers, position
encodings, feed-forward layers, trainers, schedulers, samplers, and tasks are
selected by string names.

Package areas:

- `minilab/tokenizers/`: BPE, WordPiece, Unigram, character, and byte tokenizers.
- `minilab/nn/`: attention, position encodings, normalization, feed-forward
  layers, MoE layers, residual connections, SSM blocks, diffusion blocks, and
  optimizers.
- `minilab/models/`: GPT, Mamba, Mamba-2, Hybrid, Hymba, xLSTM, ByteLatent,
  MDLM, SEDD, D3PM, and block diffusion models.
- `minilab/trainer.py`, `minilab/alignment.py`,
  `minilab/preference_alignment.py`, `minilab/online_rl.py`, and
  `minilab/diffusion_alignment.py`: pretraining, SFT, preference optimization,
  online RL, and diffusion alignment trainers.
- `minilab/data.py` and `minilab/tasks/`: dataset helpers for TinyStories,
  text8, WikiText-103, OpenWebText, Alpaca, Dolly, Anthropic HH, UltraFeedback,
  and GSM8K.
- `minilab/diffusion.py`, `minilab/diffusion_sampling.py`, and
  `minilab/generation.py`: diffusion schedules, diffusion sampling, and
  autoregressive generation.
- `minilab/evaluation.py`, `minilab/evalbench.py`, `minilab/verifiers.py`, and
  `minilab/diagnostics.py`: metrics, benchmark helpers, verifier utilities, and
  diagnostics.
- `minilab/base.py`, `minilab/config.py`, `minilab/checks.py`,
  `minilab/losses.py`, and `minilab/registry.py`: shared infrastructure.

## Scripts

- `scripts/train_tokenizer.py`: train a tokenizer implementation.
- `scripts/pretrain_lm.py`: pretrain GPT, Mamba, Mamba-2, Hybrid, Hymba, xLSTM,
  or ByteLatent models.
- `scripts/pretrain_diffusion.py`: pretrain MDLM, SEDD, D3PM, or block diffusion
  models.
- `scripts/generate.py` and `scripts/sample_diffusion.py`: sample from saved
  autoregressive and diffusion checkpoints.
- `scripts/sft.py`, `scripts/preference.py`, and `scripts/grpo.py`: run SFT,
  offline preference optimization, and online RL for autoregressive models.
- `scripts/sft_diffusion.py`, `scripts/dpo_diffusion.py`, and
  `scripts/grpo_diffusion.py`: run diffusion SFT, preference tuning, and GRPO.
- `scripts/hf_inspect.py` and `scripts/hf_generate.py`: inspect and sample
  from curated sub-1B Hugging Face causal LMs.
- `scripts/import_hf.py`: import compatible Hugging Face causal LMs into
  Minilab's native checkpoint/tokenizer format.
- `scripts/evaluate.py` and `scripts/evaluate_text8.py`: evaluate checkpoints.
- `scripts/estimate_vram.py`: estimate rough memory usage before a run.
- `scripts/compare_attention.py`, `scripts/compare_position.py`,
  `scripts/compare_connection.py`, and `scripts/compare_diffusion.py`: run
  comparison experiments.
- `scripts/common.py`: shared helpers for the script entry points.

## Dependencies

Core dependencies are `torch`, `numpy`, `regex`, `tqdm`, and `pyyaml`. Optional
extras add `datasets`, `aim`, `flash-attn`, `pytest`, and `ruff`.
</content>
</invoke>