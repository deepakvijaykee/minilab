"""Train a tokenizer on TinyStories.

    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --type unigram --vocab-size 8000 --num-texts 10000
"""

import argparse
from datasets import load_dataset
from minilab.tokenizers import available_tokenizers, build_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--type", choices=available_tokenizers(), default="bpe")
p.add_argument("--vocab-size", type=int, default=4096)
p.add_argument("--num-texts", type=int, default=5000)
p.add_argument("--save", default="tokenizer.json")
args = p.parse_args()

texts = load_dataset("roneneldan/TinyStories", split="train")["text"][: args.num_texts]
corpus = "\n".join(texts)
print(f"Corpus: {len(corpus):,} chars from {len(texts)} texts")

tok = build_tokenizer(args.type)
tok.train(corpus, vocab_size=args.vocab_size, verbose=True)
tok.save(args.save)
print(f"Saved {args.save} (vocab={tok.vocab_size})")

test = "Once upon a time there was a little girl named Lily."
ids = tok.encode(test)
decoded = tok.decode(ids)
if args.type == "wordpiece":
    print(f"  \"{test}\" -> {len(ids)} tokens, decodes as \"{decoded}\"")
else:
    if decoded != test:
        raise RuntimeError("Roundtrip failed")
    print(f"  \"{test}\" -> {len(ids)} tokens, roundtrip OK")
