"""Compare attention variants.

    python scripts/compare_attention.py --tokenizer tokenizer.json
"""

import argparse
from common import compare_lm_variants
from minilab.tokenizers import load_tokenizer
from minilab.data import load_tinystories
from minilab.trainer import TrainConfig, run_signature, set_seed

VARIANTS = [
    ("MHA", {"attention": "mha"}),
    ("MHA+QKNorm", {"attention": "mha_qknorm"}),
    ("GQA (kv=4)", {"attention": "gqa", "num_kv_heads": 4}),
    ("GQA+QKNorm", {"attention": "gqa_qknorm", "num_kv_heads": 4}),
    ("GLM partial RoPE", {"attention": "gqa_qknorm_partial_rope", "num_kv_heads": 4}),
    ("Gemma3 local/global", {"attention": "gemma3", "position": "gemma3_rope", "num_kv_heads": 4, "post_norm": True}),
    ("Gemma4 dense", {"attention": "gemma4", "position": "gemma4_rope", "num_kv_heads": 4, "ffn": "gelu_tanh", "per_layer_embedding_dim": 32, "final_logit_softcap": 30.0}),
    ("Gemma4 MoE", {"attention": "gemma4", "position": "gemma4_rope", "num_kv_heads": 4, "attention_k_eq_v": True, "ffn": "gemma4_moe", "num_experts": 8, "top_k_experts": 2}),
    ("Qwen3-Next", {"attention": "qwen3_next", "position": "qwen3_next_rope", "num_kv_heads": 2, "ffn": "gemma4_moe", "num_experts": 8, "top_k_experts": 2}),
    ("Qwen3-Coder YaRN", {"attention": "gqa_qknorm", "position": "yarn_rope", "num_kv_heads": 4, "rope_scaling_factor": 4.0}),
    ("MQA", {"attention": "mqa"}),
    ("IHA", {"attention": "iha"}),
    ("Sliding", {"attention": "sliding_window"}),
    ("BlockSparse", {"attention": "block_sparse"}),
    ("cosFormer", {"attention": "cosformer", "position": "none"}),
    ("Lightning", {"attention": "lightning", "position": "none"}),
    ("MLA", {"attention": "mla"}),
    ("CSA", {"attention": "csa"}),
    ("HCA", {"attention": "hca"}),
    ("V4-Pro", {"attention": "deepseek_v4_pro"}),
    ("V4-Flash", {"attention": "deepseek_v4_flash"}),
]

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--num-layers", type=int, default=4)
p.add_argument("--num-heads", type=int, default=8)
p.add_argument("--seq-len", type=int, default=128)
p.add_argument("--max-steps", type=int, default=2000)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--max-examples", type=int, default=10000)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
set_seed(args.seed)

tok = load_tokenizer(args.tokenizer)
train_ds = load_tinystories(tok, args.seq_len, max_examples=args.max_examples)
eval_ds = load_tinystories(tok, args.seq_len, split="validation", max_examples=1000)
tc = TrainConfig(max_steps=args.max_steps, batch_size=args.batch_size, lr=3e-4,
                 log_every=args.max_steps, eval_every=0, save_every=0, seed=args.seed)
sig = run_signature(tok, {"name": "tinystories", "split": "train", "max_examples": args.max_examples}, args.seq_len)

compare_lm_variants(
    VARIANTS,
    tok,
    train_ds,
    eval_ds,
    tc,
    sig,
    seed=args.seed,
    dim=args.dim,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    seq_len=args.seq_len,
)
