# Expected signals

The importer writes a self-contained native checkpoint directory.
Five artifacts matter and each one has a specific reason for being
there.

`model.pt`, `config.json`, and `model_type.txt` together describe the
native GPT checkpoint, with `model_type.txt` reading `GPT`. These are
the same three files that every from-scratch local-training checkpoint
contains, so the imported model loads through the unmodified native
trainers.

`tokenizer.json` plus an `hf_tokenizer/` directory hold the
tokenization state. The native `tokenizer.json` is a thin wrapper
around the underlying Hugging Face tokenizer in `hf_tokenizer/`, so
that downstream native code can call the same tokenization API
regardless of where the checkpoint came from.

`run_meta.json` records the tokenizer signature and the source
metadata for the HF model the checkpoint was imported from. The
trainer reads the signature on every subsequent run and refuses to
load a checkpoint under a mismatched tokenizer. Without that check
it would be possible to silently load a checkpoint under the wrong
tokenization and produce loss curves that look superficially
plausible but are quietly nonsensical.

`import_meta.json` records the native config and, when `VERIFY=1`,
the logit-check result. Carrying the verify result inside the
checkpoint directory means that a downstream consumer can confirm
that the import was validated without re-running the comparison.

With `VERIFY=1` the script forwards both the HF model and the native
mapped model on a short prompt and reports the maximum and mean
absolute logit difference. SmolLM2-135M usually lands `max_abs_diff`
around 1e-5. A value above 1e-3 indicates a real mapping bug, most
commonly a transposed projection or a forgotten normalization scale.
A value below 1e-5 is the floor set by fp32 accumulation order and is
not worth debugging.
