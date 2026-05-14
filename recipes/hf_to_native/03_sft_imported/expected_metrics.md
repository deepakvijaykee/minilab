# Expected signals

- The final checkpoint is
  `checkpoints/imported/<model>-sft/step_<MAX_STEPS>`, default `step_100`.
  On disk it carries `model.pt`, `config.json`, `model_type.txt`, and
  the copied tokenizer.
- `run_metrics.json` lands in the same directory and in the recipe save
  root. On CUDA it records peak PyTorch allocator memory.
- SFT loss on an already-trained 135M base is small from step 1 and
  moves slowly. The base distribution is competent; SFT here is fitting
  the Alpaca response template, which is a low-dimensional shift. Big
  jumps (loss tripling between log lines) usually mean the LR is too
  high for the imported representations and the early layers are being
  damaged. The default `2e-5` is on the conservative end for that reason.

The end-of-run generations use the same three fixed prompts as the
local-training SFT recipe. At 100 steps the result reflects the base
model with light Alpaca-flavored polish; this is a sanity run, not a
real SFT pass on a 135M model.
