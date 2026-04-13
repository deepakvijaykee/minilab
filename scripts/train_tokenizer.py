"""Train a BPE tokenizer on TinyStories.

    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 8000 --num-texts 10000
"""

import argparse
from datasets import load_dataset
from minilab.tokenizers.bpe import BPETokenizer

p = argparse.ArgumentParser()
p.add_argument("--vocab-size", type=int, default=4096)
p.add_argument("--num-texts", type=int, default=5000)
p.add_argument("--save", default="tokenizer.json")
args = p.parse_args()

texts = load_dataset("roneneldan/TinyStories", split="train")["text"][: args.num_texts]
corpus = "\n".join(texts)
print(f"Corpus: {len(corpus):,} chars from {len(texts)} texts")

tok = BPETokenizer()
tok.train(corpus, vocab_size=args.vocab_size, verbose=True)
tok.save(args.save)
print(f"Saved {args.save} (vocab={tok.vocab_size})")

test = "Once upon a time there was a little girl named Lily."
ids = tok.encode(test)
assert tok.decode(ids) == test, "Roundtrip failed"
print(f"  \"{test}\" -> {len(ids)} tokens, roundtrip OK")
