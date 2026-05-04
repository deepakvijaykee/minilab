# minilab

Minilab is a small research library for training language models from scratch,
including autoregressive LMs, text diffusion LMs, and alignment workflows.

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
- `scripts/evaluate.py` and `scripts/evaluate_text8.py`: evaluate checkpoints.
- `scripts/compare_attention.py`, `scripts/compare_position.py`,
  `scripts/compare_connection.py`, and `scripts/compare_diffusion.py`: run
  comparison experiments.
- `scripts/common.py`: shared helpers for the script entry points.

## Dependencies

Core dependencies are `torch`, `numpy`, `regex`, `tqdm`, and `pyyaml`. Optional
extras add `datasets`, `aim`, `flash-attn`, `pytest`, and `ruff`.
