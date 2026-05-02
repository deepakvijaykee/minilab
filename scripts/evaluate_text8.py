"""Evaluate a character-level LM checkpoint on the standard text8 split.

    python scripts/evaluate_text8.py --tokenizer tokenizer.char.json --checkpoint checkpoints/lm/step_50000 --split test
"""

import argparse

import torch
from torch.utils.data import DataLoader

from common import MODEL_CHOICES, load_model_checkpoint
from minilab.checks import require
from minilab.data import load_text8
from minilab.evaluation import bits_per_character
from minilab.tokenizers import load_tokenizer
from minilab.tokenizers.character import CharacterTokenizer
from minilab.trainer import validate_checkpoint_tokenizer


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="override checkpoint model family")
p.add_argument("--split", choices=["validation", "test"], default="validation")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--batch-size", type=int, default=32)
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
require(isinstance(tok, CharacterTokenizer), "text8 bits/char evaluation requires a character tokenizer")
validate_checkpoint_tokenizer(args.checkpoint, tok)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name, model = load_model_checkpoint(args.checkpoint, args.model, device=device)
model.eval()
print(f"Loaded {args.checkpoint} ({model_name}) on {device} ({model.num_parameters():,} params)")

eval_ds = load_text8(tok, args.seq_len, split=args.split, mode="lm")
bpc = bits_per_character(model, DataLoader(eval_ds, batch_size=args.batch_size))
print(f"text8 {args.split} bits/char: {bpc:.4f}")
