# minilab

Minilab is a laptop-GPU-friendly training lab for building language models end
to end: tokenizer training, tiny pretraining, supervised fine-tuning,
preference optimization, RLVR, evaluation, and diffusion LM experiments.

The point is small, faithful runs you can actually inspect, not frontier-scale
training. Everything should fit on a single device and fail loudly when
something is wrong.

## What you can train

- Tokenizers: BPE, WordPiece, Unigram, character, and byte tokenizers.
- Autoregressive LMs: GPT-style transformers, Mamba/SSM variants, hybrids,
  xLSTM, and byte-latent models.
- Diffusion LMs: MDLM, SEDD, D3PM, and block diffusion models.
- Supervised alignment: instruction SFT for autoregressive and diffusion LMs.
- Preference optimization: DPO, IPO, CPO, ORPO, SimPO, RePO, and KTO.
- RLVR and online RL: GRPO, RLOO, GSPO, DAPO, and PPO with verifier rewards.

## Laptop-scale design

- Small-model defaults that make short runs practical.
- Single-device scripts with no distributed setup required.
- Dataset caps, short sequence lengths, and small batch settings for debugging.
- Gradient accumulation, mixed precision, and gradient checkpointing where
  supported.
- Readable PyTorch implementations over framework machinery.
- Evaluation and diagnostics for comparing checkpoints and inspecting failures.

## Learning path

Start with a tokenizer, pretrain a tiny LM, then move through alignment and
evaluation:

1. Train a tokenizer.
2. Pretrain an autoregressive LM.
3. Sample from and evaluate the base checkpoint.
4. Supervised fine-tune on instruction data.
5. Run preference optimization.
6. Improve math behavior with RLVR and verifier rewards.
7. Compare autoregressive and diffusion LM training paths.

## Local training recipes

The main end-to-end path lives in `recipes/local_training/`:

Install the data extra first because the recipes load TinyStories, Alpaca,
HH-RLHF or UltraFeedback, and GSM8K through Hugging Face Datasets:

```bash
python -m pip install -e ".[data]"
```

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
bash recipes/local_training/02_sft_tiny_instruct/run.sh
bash recipes/local_training/03_preference_tiny/run.sh
bash recipes/local_training/04_grpo_tiny_math/run.sh
bash recipes/local_training/05_eval_all/run.sh
```

The diffusion branch uses the same tokenizer and mirrors the training path with
MDLM pretraining, diffusion SFT, diffusion preference optimization, and
diffusion GRPO:

```bash
bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
bash recipes/local_training/07_sft_tiny_diffusion/run.sh
bash recipes/local_training/08_preference_tiny_diffusion/run.sh
bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

The diffusion path mirrors the AR path stage for stage so the two stay
comparable, without pretending diffusion models are next-token predictors:

| Stage | AR path | Diffusion path |
| --- | --- | --- |
| Pretraining | next-token prediction | masked denoising |
| SFT | response-only label loss | prompt-fixed response denoising |
| Preference tuning | DPO/IPO/SimPO/etc. over response log-probs | DPO/VRPO over diffusion loss proxies |
| RLVR | GRPO/RLOO/etc. with generated completions | GRPO over reverse denoising trajectories |

Each recipe ships a `README.md`, `config.yaml`, `run.sh`, `expected_metrics.md`,
and `sample_outputs.md`. The defaults are scoped to fit a laptop GPU and finish
in minutes, not to chase quality. Scale up with environment overrides like
`PRESET=gpt-25m`, `MAX_STEPS=3000`, or `ALGORITHM=simpo`.

## Tiny presets

Presets remove the need to design memory-aware model sizes from scratch:

| Preset | Family | Default context | Approx params | Use case |
| --- | --- | ---: | ---: | --- |
| `gpt-10m` | GPT | 512 | ~7M-10M | default tiny training path |
| `gpt-25m` | GPT | 512 | ~20M-25M | larger SFT/preference runs |
| `gpt-60m` | GPT | 1024 | ~52M-59M | stretch local GPT runs |
| `mamba-25m` | Mamba | 512 | ~22M-29M | SSM comparison runs |
| `mdlm-25m` | MDLM | 512 | ~26M-31M | diffusion LM experiments |

Parameter counts vary with tokenizer vocabulary size. The ranges above cover
the recipe default 4k vocabulary through a 16k vocabulary.

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

There is also an optional bridge for curated sub-1B Hugging Face causal LMs:
inspect them, generate from them, or import compatible checkpoints into the
native Minilab format so they go through the same trainers as everything else.
This is not a general HF loader. Only Llama-compatible weights are wired up
today (SmolLM2 works; Qwen3 and Gemma3 are accepted by inspection and
generation but rejected by import until their weight mappings are validated).

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
`qwen3-0.6b`, plus instruct/base variants where available. The full list comes
from `scripts/hf_inspect.py --list-presets`. Recipes live in
`recipes/hf_to_native/`.

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

## Quick start

Train a tokenizer, pretrain an autoregressive LM, and sample from it:

```bash
python scripts/train_tokenizer.py --save tokenizer.json
python scripts/pretrain_lm.py --tokenizer tokenizer.json
python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
```

Run alignment on an existing checkpoint:

```bash
python scripts/sft.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
python scripts/preference.py --algorithm dpo --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
```

Train and sample a diffusion LM:

```bash
python scripts/pretrain_diffusion.py --tokenizer tokenizer.json --model mdlm
python scripts/sample_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
```

Run diffusion alignment on an existing diffusion checkpoint:

```bash
python scripts/sft_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
python scripts/dpo_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion_sft/step_3000
python scripts/grpo_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion_sft/step_3000
```

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
