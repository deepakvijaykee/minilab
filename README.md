# Minilab

Minilab is a laptop-GPU-friendly lab for training and post-training compact GPT language models. The project keeps one model family and exposes research variation through reusable Transformer components rather than separate architecture stacks.

## What is included

- Character, byte, BPE, unigram, WordPiece, and Hugging Face tokenizers.
- GPT pretraining with configurable attention, position encoding, normalization, FFN/MoE, residual connections, and optimizers.
- Supervised fine-tuning and preference methods including DPO, IPO, CPO, ORPO, RePO, SimPO, and KTO.
- Online reinforcement-learning methods including PPO, GRPO, Dr.GRPO, GSPO, RLOO, DAPO, TPO, VPO, and related research lanes.
- Rule-based and learned verifiers, evaluation utilities, diagnostics, and checkpoint metadata validation.
- Hugging Face import and native post-training recipes.

## Install

```bash
python -m pip install -e ".[data,dev]"
```

PyTorch is the only heavyweight required dependency. Optional extras are available for Hugging Face integration, logging, and FlashAttention.

## Local end-to-end workflow

Each recipe is self-contained and can be run from the repository root:

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
bash recipes/local_training/02_sft_tiny_instruct/run.sh
bash recipes/local_training/03_preference_tiny/run.sh
bash recipes/local_training/04_grpo_tiny_math/run.sh
bash recipes/local_training/05_eval_all/run.sh
```

The defaults are intentionally small. Override recipe environment variables such as `MAX_STEPS`, `BATCH_SIZE`, `SEQ_LEN`, and `PRESET` when you want a longer run.

Available presets:

| Preset | Typical role |
|---|---|
| `gpt-10m` | quick local iteration |
| `gpt-25m` | larger laptop experiments |
| `gpt-60m` | stretch runs with more headroom |

## Direct CLI examples

Train a tokenizer and model:

```bash
python scripts/train_tokenizer.py --dataset tinystories --output tokenizer.json
python scripts/pretrain_lm.py --tokenizer tokenizer.json --preset gpt-10m
```

Fine-tune and run preference optimization:

```bash
python scripts/sft.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
python scripts/preference.py --algorithm dpo --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
```

Run verifier-guided training and generation:

```bash
python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/grpo/step_1000 --prompt "Solve 12 + 30."
```

Inspect or import a Hugging Face model:

```bash
python scripts/hf_inspect.py --model HuggingFaceTB/SmolLM2-135M
python scripts/import_hf.py --model HuggingFaceTB/SmolLM2-135M --output checkpoints/imported/smollm2-135m
```

See `recipes/hf_to_native/` for the complete import, SFT, preference, and GRPO sequence.

## GPT component experiments

The single GPT stack supports controlled experiments without multiplying model families:

- Attention: MHA, MQA, GQA, sparse/local/compressed variants, latent attention, and recurrent attention variants.
- Position: learned, sinusoidal, RoPE, YaRN, ALiBi, T5 relative bias, and related methods.
- FFN: dense gated FFNs and several MoE routing strategies.
- Connections: residual, hyperconnection, manifold hyperconnection, and research variants.
- Training: AdamW, Lion, Muon, soft-Muon, checkpointing, MTP, LayerSkip, and auxiliary objectives.

Comparison scripts under `scripts/compare_*.py` run matched GPT experiments for individual component axes.

## Project layout

- `minilab/models/gpt.py`: the model and its configuration contract.
- `minilab/nn/`: reusable attention, position, normalization, FFN, MoE, connection, and optimizer components.
- `minilab/trainer.py`: pretraining loop, checkpointing, resume validation, and optimizer wiring.
- `minilab/preference_alignment.py` and `minilab/online_rl.py`: post-training algorithms.
- `minilab/data.py`: language-model, SFT, preference, verifier, and evaluation datasets.
- `minilab/generation.py`: autoregressive and verified decoding.
- `scripts/`: runnable training, evaluation, import, diagnostics, and comparison entry points.
- `recipes/`: reproducible local and Hugging Face workflows.
- `tests/`: unit, invariant, integration, and script-contract coverage.

## Verification

```bash
python -m pytest -q
python -m compileall -q minilab scripts
```

Checkpoints and local caches are intentionally excluded from version control.
