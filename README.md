# minilab

A modular library for training language models from scratch: autoregressive transformers, text diffusion, and alignment.

Built for learning and experimentation. Every component is swappable via a registry. No HuggingFace transformers/tokenizers dependency, all of models, tokenizers, trainers all written from scratch.

## Install

```bash
conda create -n minilab python=3.12 -y
conda activate minilab
pip install torch numpy regex tqdm pyyaml datasets
pip install -e .
```

## Quick start

```bash
# 1. Train a BPE tokenizer on TinyStories
python scripts/train_tokenizer.py --save tokenizer.json

# 2. Pretrain GPT
python scripts/pretrain_lm.py --tokenizer tokenizer.json

# 3. Generate text
python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000

# 4. SFT on Alpaca
python scripts/sft.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000

# 5. DPO on Anthropic HH
python scripts/dpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000

# 6. GRPO on GSM8K
python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
```

## Swappable components

Every architectural choice is a registry entry. Swap via config strings:

```python
from minilab.models.gpt import GPT, GPTConfig

model = GPT(GPTConfig(
    vocab_size=4096, dim=256, num_layers=6, num_heads=8,
    attention="iha",       # "mha", "gqa", "iha"
    position="rope",       # "rope", "alibi", "learned", "sinusoidal"
    norm="rmsnorm",        # "rmsnorm", "layernorm"
    ffn="swiglu",          # "swiglu", "gelu", "moe"
    connection="residual", # "residual", "hc", "mhc"
))
```

Or use a preset:

```python
from minilab.models.gpt import gpt_preset
model = GPT(gpt_preset("gpt-small", vocab_size=4096))
```

## What's in the library

### Tokenizers (from scratch, no dependencies)

| Algorithm | Used by | File |
|-----------|---------|------|
| BPE | GPT, Llama, Mistral | `tokenizers/bpe.py` |
| WordPiece | BERT, Electra | `tokenizers/wordpiece.py` |
| Unigram | T5, XLNet, ALBERT | `tokenizers/unigram.py` |
| Character | text8 experiments | `tokenizers/character.py` |

### Neural network primitives

| Category | Options |
|----------|---------|
| Attention | MHA, GQA, IHA (cross-head mixing) |
| Position | RoPE, ALiBi, learned, sinusoidal |
| Normalization | RMSNorm, LayerNorm |
| Feed-forward | SwiGLU, GELU, MoE (sparse mixture of experts with load balancing) |
| Connections | Residual, HC (hyper-connections), mHC (manifold-constrained) |

### Models

| Model | Type | Paper |
|-------|------|-------|
| GPT | Autoregressive | Configurable with all components above |
| MDLM | Masked diffusion | Sahoo et al., NeurIPS 2024 |
| SEDD | Score entropy diffusion | Lou et al., ICML 2024 |
| D3PM | Discrete denoising diffusion | Austin et al., NeurIPS 2021 |

### Training

| Trainer | Purpose |
|---------|---------|
| LMTrainer | Next-token prediction |
| DiffusionTrainer | Denoising (works for MDLM, SEDD, D3PM) |
| SFTTrainer | Supervised fine-tuning |
| DPOTrainer | Direct Preference Optimization |
| GRPOTrainer | Group Relative Policy Optimization |

Features: AdamW/Lion/Muon optimizers, cosine/linear/constant/WSD LR schedules, mixed precision, gradient accumulation, gradient checkpointing, torch.compile, aim logging, seed management.

### Data loaders

| Dataset | Use | Function |
|---------|-----|----------|
| TinyStories | Pretraining | `load_tinystories()` |
| text8 | Character-level | `load_text8()` |
| WikiText-103 | Pretraining | `load_wikitext()` |
| OpenWebText | Pretraining | `load_openwebtext()` |
| Alpaca | SFT | `load_alpaca()` |
| Dolly-15k | SFT | `load_dolly()` |
| Anthropic HH | DPO | `load_hh_rlhf()` |
| UltraFeedback | DPO | `load_ultrafeedback()` |
| GSM8K | GRPO | `load_gsm8k()` |

### Generation

- Autoregressive: top-k, top-p, temperature, repetition penalty, greedy
- Diffusion ancestral: iterative unmasking from full mask
- Diffusion DDPM cache: cached predictions for ~4x fewer forward passes
- Infilling: regenerate masked spans while keeping context (unique to diffusion)

### Evaluation

- Perplexity, distinct-n, self-BLEU, accuracy reward, format reward, number extraction

## Scripts

Each script does one thing. They chain via `--tokenizer` and `--checkpoint` flags.

```
train_tokenizer.py      ->  tokenizer.json
pretrain_lm.py          ->  checkpoints/lm/
pretrain_diffusion.py   ->  checkpoints/diffusion/
sft.py                  ->  checkpoints/sft/
dpo.py                  ->  checkpoints/dpo/
grpo.py                 ->  checkpoints/grpo/
generate.py                 (reads AR checkpoint)
sample_diffusion.py         (reads diffusion checkpoint)
evaluate.py                 (perplexity + diversity metrics)
compare_attention.py        (MHA vs GQA vs IHA)
compare_position.py         (RoPE vs ALiBi vs learned vs sinusoidal)
compare_connection.py       (Residual vs HC vs mHC)
compare_diffusion.py        (MDLM vs SEDD vs D3PM)
```

## Architecture

```
minilab/
├── registry.py          # Decorator-based component registration
├── config.py            # BaseConfig with JSON serialization
├── base.py              # BaseModel (save/load/grad ckpt), BaseTokenizer
├── nn/
│   ├── attention.py     # MHA, GQA, IHA
│   ├── position.py      # RoPE, ALiBi, learned, sinusoidal
│   ├── norm.py          # RMSNorm, LayerNorm
│   ├── ffn.py           # SwiGLU, GELU
│   ├── moe.py           # Sparse MoE with load balancing
│   ├── connections.py   # Residual, HC, mHC (Sinkhorn-constrained)
│   ├── diffusion.py     # AdaLN, DiffusionBlock, SinusoidalTimeEmbedding
│   └── optimizers.py    # Muon, Lion
├── tokenizers/
│   ├── bpe.py
│   ├── wordpiece.py
│   ├── unigram.py
│   └── character.py
├── models/
│   ├── gpt.py           # GPT + TransformerBlock + presets
│   ├── mdlm.py          # Masked Diffusion LM
│   ├── sedd.py          # Score Entropy Discrete Diffusion
│   └── d3pm.py          # Discrete Denoising Diffusion
├── data.py              # Datasets + loaders (TinyStories, Alpaca, HH, GSM8K, text8, WikiText)
├── diffusion.py         # ForwardProcess + noise schedules (cosine, linear, log-linear)
├── trainer.py           # Trainer, LMTrainer, DiffusionTrainer
├── alignment.py         # SFTTrainer, DPOTrainer, GRPOTrainer
├── generation.py        # AR sampling, diffusion sampling (ancestral, DDPM cache), infilling
└── evaluation.py        # Perplexity, distinct-n, self-BLEU, rewards
```

## Dependencies

Core: `torch`, `numpy`, `regex`, `tqdm`, `pyyaml`. Optional: `datasets` (for data loaders), `aim` (for logging).

No dependency on HuggingFace transformers, tokenizers, sentencepiece, or trl.
