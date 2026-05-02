"""Train a tokenizer on a supported pretraining corpus.

    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --type unigram --vocab-size 8000 --num-texts 10000
    python scripts/train_tokenizer.py --dataset text8 --type character --num-texts 0 --save tokenizer.char.json
"""

import argparse

from common import PRETRAIN_DATASET_CHOICES
from minilab.checks import require
from minilab.data import _example_limit, load_dataset, text8_standard_split
from minilab.tokenizers import available_tokenizers, build_tokenizer
from minilab.tokenizers.byte import BYTE_VOCAB_SIZE

_VERBOSE_TOKENIZERS = {"bpe", "unigram", "wordpiece"}


def _load_tokenizer_corpus(dataset, num_texts):
    if dataset == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = ds["text"][:_example_limit(num_texts, len(ds))]
        return "\n".join(texts), f"{len(texts)} TinyStories rows"
    if dataset == "text8":
        require(num_texts == 0, "--num-texts does not apply to text8; pass 0 to use the standard train split")
        text = text8_standard_split(load_dataset("afmck/text8", split="train")[0]["text"], "train")
        return text, "the standard text8 train split"
    if dataset == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        rows = ds["text"][:_example_limit(num_texts, len(ds))]
        texts = [text for text in rows if len(text) > 50]
        return "\n".join(texts), f"{len(texts)} WikiText rows"
    if dataset == "openwebtext":
        require(num_texts > 0, "streaming OpenWebText tokenizer training requires --num-texts > 0")
        texts = []
        for row in load_dataset("Skylion007/openwebtext", split="train", streaming=True):
            texts.append(row["text"])
            if len(texts) >= num_texts:
                break
        return "\n".join(texts), f"{len(texts)} OpenWebText rows"
    raise ValueError(f"Unknown tokenizer dataset: {dataset}")


p = argparse.ArgumentParser()
p.add_argument("--type", choices=available_tokenizers(), default="bpe")
p.add_argument("--dataset", choices=PRETRAIN_DATASET_CHOICES, default=None)
p.add_argument("--vocab-size", type=int, default=None)
p.add_argument("--num-texts", type=int, default=None)
p.add_argument("--save", default="tokenizer.json")
args = p.parse_args()

if args.type == "byte":
    require(args.dataset is None, "--dataset does not apply to the fixed byte tokenizer")
    require(args.num_texts is None, "--num-texts does not apply to the fixed byte tokenizer")
    require(args.vocab_size in {None, 0, BYTE_VOCAB_SIZE}, (
        f"--vocab-size for the fixed byte tokenizer must be 0 or {BYTE_VOCAB_SIZE}"
    ))
    dataset = "fixed-byte"
    num_texts = 0
    vocab_size = 0 if args.vocab_size is None else args.vocab_size
    corpus = ""
    units = "the fixed UTF-8 byte vocabulary"
else:
    dataset = args.dataset or "tinystories"
    vocab_size = 4096 if args.vocab_size is None else args.vocab_size
    if dataset == "text8":
        num_texts = 0 if args.num_texts is None else args.num_texts
    else:
        num_texts = 5000 if args.num_texts is None else args.num_texts
    require(num_texts >= 0, "--num-texts must be >= 0")
    corpus, units = _load_tokenizer_corpus(dataset, num_texts)
print(f"Corpus: {len(corpus):,} chars from {units}")

tok = build_tokenizer(args.type)
if args.type in _VERBOSE_TOKENIZERS:
    tok.train(corpus, vocab_size=vocab_size, verbose=True)
else:
    tok.train(corpus, vocab_size=vocab_size)
tok.save(args.save)
print(f"Saved {args.save} (vocab={tok.vocab_size})")

test = "once upon a time there was a little girl named lily" if dataset == "text8" else "Once upon a time there was a little girl named Lily."
ids = tok.encode(test)
decoded = tok.decode(ids)
if args.type == "wordpiece":
    print(f"  \"{test}\" -> {len(ids)} tokens, decodes as \"{decoded}\"")
else:
    if decoded != test:
        raise RuntimeError("Roundtrip failed")
    print(f"  \"{test}\" -> {len(ids)} tokens, roundtrip OK")
