# Expected signals

The final checkpoint is
`checkpoints/imported/<model>-sft/step_<MAX_STEPS>`, which defaults to
`step_100`. On disk it carries `model.pt`, `config.json`,
`model_type.txt`, and the copied tokenizer, matching the layout of
any from-scratch SFT checkpoint produced by the local-training recipes.
The `run_metrics.json` file is written to the same directory and also in
the recipe save root, and on CUDA it records peak PyTorch allocator
memory, the number to size against before scaling the run up.

SFT loss on an already-trained 135M base starts small and moves
slowly. The base distribution is already competent over a broad text
corpus, so what SFT is doing here is fitting the Alpaca response
template, which is a small shift relative to the base. A
sudden large jump, for example loss tripling between two log lines,
usually indicates that the learning rate is too high for the
imported representations and the early layers are being damaged
rather than refined. The default of 2e-5 sits on the conservative
end of the useful range for exactly that reason.

The end-of-run generations reuse the same three fixed prompts as the
local-training SFT recipe, so the imported and from-scratch runs can be
read side by side, with matched prompts keeping the comparison honest. At a
hundred steps the result reflects the base model with light Alpaca-
flavored polish. A hundred steps is enough to confirm the import path
trains cleanly. A real SFT pass on a 135M base would run far longer.
