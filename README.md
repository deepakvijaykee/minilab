# minilab

A modular library for training language models from scratch: autoregressive transformers, text diffusion, and alignment.

Built for learning and experimentation. GPT components, tokenizers, schedulers, samplers, and trainers are registered extension points; diffusion models share a compact fixed transformer backbone so each objective can keep its reverse-process contract explicit. No HuggingFace transformers/tokenizers dependency, all of models, tokenizers, trainers all written from scratch.

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

# 2. Pretrain an autoregressive LM
python scripts/pretrain_lm.py --tokenizer tokenizer.json
#    Or train a Mamba SSM LM
python scripts/pretrain_lm.py --tokenizer tokenizer.json --model mamba

# 3. Generate text
python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000

# 4. SFT on Alpaca
python scripts/sft.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000

# 5. DPO on Anthropic HH
python scripts/preference.py --algorithm dpo --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000

# 6. GRPO on GSM8K
python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
```

## Extension Points

GPT architectural choices are registry entries. Swap via config strings:

```python
from minilab.models.gpt import GPT, GPTConfig

model = GPT(GPTConfig(
    vocab_size=4096, dim=256, num_layers=6, num_heads=8,
    attention="gemma4",    # "mha", "qwen3_next", "lightning", "gemma4", "mla", ...
    position="gemma4_rope", # "rope", "yarn_rope", "gemma4_rope", "alibi", ...
    norm="rmsnorm",        # "rmsnorm", "layernorm"
    ffn="gelu_tanh",       # "swiglu", "geglu", "gelu_tanh", "gemma4_moe", ...
    connection="residual", # "residual", "hc", "mhc"
))
```

Or use a preset:

```python
from minilab.models.gpt import gpt_preset
model = GPT(gpt_preset("gpt-small", vocab_size=4096))
```

For diffusion experiments, add new algorithms by registering a model with a
`forward_process_type` and `reverse_parameterization`, then pair it with a sampler
that validates the parameterization it consumes. Schedulers are registry entries;
continuous-time objectives also require the schedule to register a continuous
`alpha(t)` function and its exact derivative. The current MDLM, SEDD, and D3PM
implementations keep their transformer block fixed to make objective differences
easy to inspect.

## What's in the library

### Tokenizers (from scratch, no dependencies)

| Algorithm | Used by | File |
|-----------|---------|------|
| BPE | GPT, Llama, Mistral | `tokenizers/bpe.py` |
| WordPiece | BERT, Electra; whitespace-normalizing, not reversible | `tokenizers/wordpiece.py` |
| Unigram | T5, XLNet, ALBERT | `tokenizers/unigram.py` |
| Character | text8 experiments | `tokenizers/character.py` |

### Neural network primitives

| Category | Options |
|----------|---------|
| Attention | MHA, MQA, GQA, QK-Norm MHA/GQA, gated Qwen3-Next GQA, Gated DeltaNet, Lightning Attention-2 reference path, Gemma3/Gemma4 local-global schedules, GLM partial-RoPE GQA, IHA, sliding-window, block-sparse, cosFormer, MLA, CSA/HCA, DeepSeek V4 schedules |
| Position | RoPE with configurable base, Gemma3 local/global RoPE bases, Gemma4 proportional RoPE, YaRN RoPE scaling, ALiBi, T5 relative bias, KERPLE log/power, learned, sinusoidal, NoPE |
| Normalization | RMSNorm, LayerNorm |
| Feed-forward | SwiGLU, GEGLU, ReGLU, GELU, GELU-tanh, token-choice/Mixtral/Switch/expert-choice/DeepSeek/aux-free/BASE/Gemma4 MoE |
| Connections | Residual, dynamic HC, mHC with Sinkhorn-constrained residual maps |
| SSM | Mamba selective state-space LM |

### Models

| Model | Type | Paper |
|-------|------|-------|
| GPT | Autoregressive | Configurable with all components above |
| MambaLM | Autoregressive SSM | Gu and Dao, 2023 |
| MDLM | Masked diffusion | Sahoo et al., NeurIPS 2024 |
| SEDD | Score entropy diffusion with absorbing graph | Lou et al., ICML 2024 |
| D3PM | Absorbing discrete denoising diffusion | Austin et al., NeurIPS 2021 |

### Training

| Trainer | Purpose |
|---------|---------|
| LMTrainer | Next-token prediction |
| DiffusionTrainer | Denoising (works for MDLM, SEDD, D3PM) |
| SFTTrainer | Supervised fine-tuning |
| DPOTrainer | Direct Preference Optimization |
| IPOTrainer | Identity Preference Optimization |
| CPOTrainer / SimPOTrainer / ORPOTrainer | Reference-free preference optimization |
| KTOTrainer | Binary desirable/undesirable alignment |
| PPOTrainer | Actor-critic RLHF/RLVR with learned value head |
| GRPOTrainer / RLOOTrainer | Critic-free online RLHF/RLVR policy gradients |
| GSPOTrainer / DAPOTrainer | Recent sequence-level and long-CoT RLVR variants |
| DiffusionSFTTrainer | Response-only diffusion SFT |
| DiffusionDPOTrainer | Preference tuning with diffusion loss proxy |
| DiffusionGRPOTrainer | Reverse-trajectory GRPO for diffusion LMs |

Features: AdamW/Lion/Muon optimizers, MuonClip-style QK-Clip update hook, cosine/linear/constant/WSD LR schedules, mixed precision, gradient accumulation, gradient checkpointing, torch.compile, aim logging, seed management.

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
| Anthropic HH | KTO | `load_hh_rlhf_kto()` |
| UltraFeedback | Preference/KTO | `load_ultrafeedback()`, `load_ultrafeedback_kto()` |
| GSM8K | GRPO | `load_gsm8k()` |

### Generation

- Autoregressive: top-k, top-p, temperature, repetition penalty, greedy
- Diffusion ancestral: iterative unmasking from full mask
- Diffusion DDPM cache: cached predictions for ~4x fewer forward passes
- SEDD analytical and D3PM x0-parameterized samplers preserve each model's reverse-process contract
- Infilling: regenerate masked spans while keeping context, dispatched by diffusion parameterization

### Evaluation

- Perplexity, distinct-n, self-BLEU, accuracy reward, format reward, number extraction

## Scripts

Each script does one thing. They chain via `--tokenizer` and `--checkpoint` flags.

```
train_tokenizer.py      ->  tokenizer.json
pretrain_lm.py          ->  checkpoints/lm/
pretrain_diffusion.py   ->  checkpoints/diffusion/
sft.py                  ->  checkpoints/sft/
preference.py           ->  checkpoints/{dpo,ipo,cpo,simpo,orpo,kto}/
grpo.py                 ->  checkpoints/grpo/      (PPO/GRPO/DAPO/GSPO/RLOO via --algorithm)
sft_diffusion.py        ->  checkpoints/diffusion_sft/
dpo_diffusion.py        ->  checkpoints/diffusion_dpo/
grpo_diffusion.py       ->  checkpoints/diffusion_grpo/
generate.py                 (reads AR checkpoint)
sample_diffusion.py         (reads diffusion checkpoint)
evaluate.py                 (perplexity + diversity metrics)
compare_attention.py        (MHA/GQA/QK-Norm/Gemma3/Gemma4/Qwen3-Next/Lightning/MLA/CSA/HCA)
compare_position.py         (RoPE/Gemma4 p-RoPE/YaRN/ALiBi/T5/KERPLE/learned/sinusoidal/NoPE)
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
│   ├── attention.py     # MHA/MQA/GQA/QK-Norm, Gemma4/Qwen3-Next/Lightning, sparse, MLA, CSA/HCA
│   ├── position.py      # RoPE, Gemma4 proportional RoPE, YaRN, ALiBi, T5/KERPLE, learned, none
│   ├── ssm.py           # Mamba selective scan reference path
│   ├── norm.py          # RMSNorm, LayerNorm
│   ├── ffn.py           # SwiGLU, GEGLU, ReGLU, GELU, GELU-tanh
│   ├── moe.py           # Sparse, shared, aux-free, expert-choice, BASE, Gemma4 MoE
│   ├── connections.py   # Residual, dynamic HC, mHC (Sinkhorn-constrained)
│   ├── diffusion.py     # AdaLN, DiffusionBlock, SinusoidalTimeEmbedding
│   └── optimizers.py    # Muon, Lion
├── tokenizers/
│   ├── bpe.py
│   ├── wordpiece.py
│   ├── unigram.py
│   └── character.py
├── models/
│   ├── gpt.py           # GPT + TransformerBlock + presets
│   ├── mamba.py         # MambaLM autoregressive SSM
│   ├── mdlm.py          # Masked Diffusion LM
│   ├── sedd.py          # Score Entropy Discrete Diffusion
│   └── d3pm.py          # Discrete Denoising Diffusion
├── data.py              # Datasets + loaders (TinyStories, Alpaca, HH, GSM8K, text8, WikiText)
├── diffusion.py         # ForwardProcess + noise schedules (cosine, linear, geometric, log-linear)
├── trainer.py           # Trainer, LMTrainer, DiffusionTrainer
├── alignment.py         # AR and diffusion SFT/DPO/GRPO trainers
├── generation.py        # AR sampling, diffusion sampling (ancestral, DDPM cache), infilling
└── evaluation.py        # Perplexity, distinct-n, self-BLEU, rewards
```

## Dependencies

Core: `torch`, `numpy`, `regex`, `tqdm`, `pyyaml`. Optional: `datasets` (for data loaders), `aim` (for logging).

No dependency on HuggingFace transformers, tokenizers, sentencepiece, or trl.
