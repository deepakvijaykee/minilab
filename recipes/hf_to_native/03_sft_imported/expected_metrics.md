# Expected signals

- `checkpoints/imported/<model>-sft/step_<MAX_STEPS>` is the final
  checkpoint (default `step_100`). On disk it carries `model.pt`,
  `config.json`, `model_type.txt`, and the copied tokenizer.
- `run_metrics.json` lands in the same directory and in the recipe save
  root. On CUDA it includes peak PyTorch allocator memory.
- SFT loss on an already-trained 135M base is small from step 1 and moves
  slowly. Big jumps (loss tripling between log lines) usually mean the
  learning rate is too high for the imported representations. The
  default `2e-5` is on the conservative end for that reason.

The end-of-run generation block uses the same three fixed prompts as the
local-training SFT recipe. At a 100-step run the quality reflects the
base model with light Alpaca-flavored polish, not a real SFT pass.
