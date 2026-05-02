# minilab

A modular library for training language models from scratch: autoregressive transformers, text diffusion, and alignment.

Built for learning and experimentation. GPT components, tokenizers, schedulers, samplers, and trainers are registered extension points; diffusion models share a compact fixed transformer backbone so each objective can keep its reverse-process contract explicit. No HuggingFace transformers/tokenizers dependency, all of models, tokenizers, trainers all written from scratch. Paper-named components are documented by fidelity level in [`docs/implementation_fidelity.md`](docs/implementation_fidelity.md): some are direct reference paths, some are deliberately scoped variants, and some are architecture-inspired research knobs. The current smoke and long-run validation plan lives in [`docs/experiment_matrix.md`](docs/experiment_matrix.md); the latest module-by-module fidelity audit is in [`docs/fidelity_audit.md`](docs/fidelity_audit.md).

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
#    Or train Mamba-2 / Hymba-style variants
python scripts/pretrain_lm.py --tokenizer tokenizer.json --model mamba2
python scripts/pretrain_lm.py --tokenizer tokenizer.json --model hymba

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
    norm="rmsnorm",        # "rmsnorm", "zero_centered_rmsnorm", "layernorm"
    ffn="gelu_tanh",       # "swiglu", "geglu", "qwen3_next_moe", "gemma4_moe", ...
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
| Byte | BLT-style byte latent experiments | `tokenizers/byte.py` |

### Neural network primitives

| Category | Options |
|----------|---------|
| Attention | MHA, MQA, GQA, QK-Norm MHA/GQA, Qwen3-Next-style gated GQA and Gated DeltaNet reference paths, Lightning Attention-2 reference path, Gemma-style local/global schedules, GLM-style partial-RoPE GQA, IHA, sliding-window, block-sparse, cosFormer, MLA, CSA/HCA, DeepSeek-V4-style compressed schedules |
| Position | RoPE with configurable base, Gemma-style local/global RoPE bases and proportional RoPE, YaRN RoPE scaling, ALiBi, T5 relative bias, KERPLE log/power, learned, sinusoidal, NoPE |
| Normalization | RMSNorm, zero-centered RMSNorm, LayerNorm |
| Feed-forward | SwiGLU, GEGLU, ReGLU, GELU, GELU-tanh, token-choice/Mixtral/Switch/expert-choice/DeepSeek-style/Qwen3-Next-style/aux-free/BASE/Gemma-style MoE |
| Connections | Residual, dynamic HC, mHC with Sinkhorn-constrained residual maps |
| SSM | Mamba selective scan and Mamba-2 SSD reference paths |

### Models

| Model | Type | Paper |
|-------|------|-------|
| GPT | Autoregressive | Configurable with all components above |
| MambaLM | Autoregressive SSM | Gu and Dao, 2023 |
| Mamba2LM | Autoregressive SSM | Dao and Gu, 2024 Mamba-2/SSD |
| HybridLM | Autoregressive attention/SSM hybrid | Jamba-style interleaving |
| HymbaLM | Autoregressive hybrid-head LM | Hymba-style parallel attention/SSM with optional meta tokens |
| XLSTMLM | Autoregressive mLSTM stack | Beck et al., 2024/2025 xLSTM Large-style block |
| MDLM | Masked diffusion | Sahoo et al., NeurIPS 2024 |
| SEDD | Score entropy diffusion with absorbing graph | Lou et al., ICML 2024 |
| D3PM | Absorbing discrete denoising diffusion | Austin et al., NeurIPS 2021 |
| BlockDiffusionLM | Block masked diffusion | BD3-style block diffusion |
| ByteLatentLM | Byte latent autoregressive LM | BLT-style local/global byte modeling |
| OutcomeVerifier | Learned verifier | VerIF/RLVR-style outcome supervision |

### Training

| Trainer | Purpose |
|---------|---------|
| LMTrainer | Next-token prediction |
| DiffusionTrainer | Denoising (works for MDLM, SEDD, D3PM) |
| SFTTrainer | Supervised fine-tuning |
| DPOTrainer | Direct Preference Optimization |
| IPOTrainer | Identity Preference Optimization |
| CPOTrainer / SimPOTrainer / ORPOTrainer | Reference-free preference optimization |
| RePOTrainer | ReLU-based Preference Optimization |
| KTOTrainer | Binary desirable/undesirable alignment |
| PPOTrainer | Actor-critic RLHF/RLVR with learned value head |
| GRPOTrainer / RLOOTrainer | Critic-free online RLHF/RLVR policy gradients |
| GSPOTrainer / DAPOTrainer | Recent sequence-level and long-CoT RLVR variants |
| DiffusionSFTTrainer | Response-only diffusion SFT |
| DiffusionDPOTrainer | Preference tuning with diffusion loss proxy |
| DiffusionVRPOTrainer | Variance-reduced diffusion preference tuning |
| DiffusionGRPOTrainer | Reverse-trajectory GRPO for diffusion LMs |

Features: AdamW/Lion/Muon optimizers, MuonClip-style QK-Clip update hook, cosine/linear/constant/WSD LR schedules, mixed precision, gradient accumulation, gradient checkpointing, torch.compile, aim logging, seed management.

### Data loaders

| Dataset | Use | Function |
|---------|-----|----------|
| TinyStories | Pretraining | `load_tinystories()` |
| text8 | Character-level, standard 90M/5M/5M split | `load_text8()` |
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
- Diffusion semi-autoregressive: extend prompts by denoising masked blocks with previous blocks fixed
- Dream-style masked generation: origin, MaskGIT confidence, top-k margin, and entropy remasking policies
- Infilling: regenerate masked spans while keeping context, dispatched by diffusion parameterization

### Evaluation

- Perplexity, text8 bits/character, distinct-n, self-BLEU, accuracy reward, format reward, number extraction
- RULER synthetic task keys, token-budget helpers, JSONL rows, and containment metrics
- English LongBench prompt templates, dataset metric map, max generation lengths, and max-over-gold scorer
- Rule, composite, tool-call, and learned outcome verifier utilities

## Scripts

Each script does one thing. They chain via `--tokenizer` and `--checkpoint` flags.

```
train_tokenizer.py      ->  tokenizer.json
pretrain_lm.py          ->  checkpoints/lm/
pretrain_diffusion.py   ->  checkpoints/diffusion/
sft.py                  ->  checkpoints/sft/
preference.py           ->  checkpoints/{dpo,ipo,cpo,simpo,orpo,repo,kto}/
grpo.py                 ->  checkpoints/grpo/      (PPO/GRPO/DAPO/GSPO/RLOO via --algorithm)
sft_diffusion.py        ->  checkpoints/diffusion_sft/
dpo_diffusion.py        ->  checkpoints/diffusion_dpo/ (DPO or VRPO)
grpo_diffusion.py       ->  checkpoints/diffusion_grpo/
generate.py                 (reads AR checkpoint)
sample_diffusion.py         (reads diffusion checkpoint)
evaluate.py                 (perplexity + diversity metrics)
evaluate_text8.py           (character-level text8 bits/character)
compare_attention.py        (MHA/GQA/QK-Norm/Gemma-style/Qwen3-Next-style/Lightning/MLA/CSA/HCA)
compare_position.py         (RoPE/proportional RoPE/YaRN/ALiBi/T5/KERPLE/learned/sinusoidal/NoPE)
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
│   ├── attention.py     # MHA/MQA/GQA/QK-Norm, Gemma/Qwen/Lightning-style paths, sparse, MLA, CSA/HCA
│   ├── position.py      # RoPE, proportional RoPE, YaRN, ALiBi, T5/KERPLE, learned, none
│   ├── ssm.py           # Mamba selective scan and Mamba-2 SSD reference paths
│   ├── norm.py          # RMSNorm, zero-centered RMSNorm, LayerNorm
│   ├── ffn.py           # SwiGLU, GEGLU, ReGLU, GELU, GELU-tanh
│   ├── moe.py           # Sparse, shared, aux-free, expert-choice, BASE, Gemma-style MoE
│   ├── connections.py   # Residual, dynamic HC, mHC (Sinkhorn-constrained)
│   ├── diffusion.py     # AdaLN, DiffusionBlock, SinusoidalTimeEmbedding
│   └── optimizers.py    # Muon, Lion
├── tokenizers/
│   ├── bpe.py
│   ├── byte.py
│   ├── wordpiece.py
│   ├── unigram.py
│   └── character.py
├── models/
│   ├── gpt.py           # GPT + TransformerBlock + presets
│   ├── hybrid.py        # Attention/Mamba interleaved LM
│   ├── hymba.py         # Hymba-style parallel attention/SSM LM
│   ├── mamba.py         # MambaLM autoregressive SSM
│   ├── mamba2.py        # Mamba2LM SSD autoregressive SSM
│   ├── byte_latent.py   # BLT-style byte latent LM
│   ├── block_diffusion.py # BD3-style block diffusion LM
│   ├── mdlm.py          # Masked Diffusion LM
│   ├── sedd.py          # Score Entropy Discrete Diffusion
│   └── d3pm.py          # Discrete Denoising Diffusion
├── data.py              # Datasets + loaders (TinyStories, Alpaca, HH, GSM8K, text8, WikiText, OpenWebText)
├── evalbench.py         # RULER and English LongBench helpers
├── diffusion.py         # ForwardProcess + noise schedules (cosine, linear, geometric, log-linear)
├── trainer.py           # Trainer, LMTrainer, DiffusionTrainer
├── alignment.py         # AR and diffusion SFT/DPO/GRPO trainers
├── generation.py        # AR sampling, diffusion sampling (ancestral, DDPM cache), infilling
├── evaluation.py        # Perplexity, bits/char, distinct-n, self-BLEU, rewards
└── verifiers.py         # Rule and learned verifier utilities
```

## Dependencies

Core: `torch`, `numpy`, `regex`, `tqdm`, `pyyaml`. Optional: `datasets` (for data loaders), `aim` (for logging).

No dependency on HuggingFace transformers, tokenizers, sentencepiece, or trl.
