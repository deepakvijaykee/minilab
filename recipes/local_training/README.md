# Local training recipes

These recipes form one end-to-end GPT workflow. Run them from the repository root.

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
bash recipes/local_training/02_sft_tiny_instruct/run.sh
bash recipes/local_training/03_preference_tiny/run.sh
bash recipes/local_training/04_grpo_tiny_math/run.sh
bash recipes/local_training/05_eval_all/run.sh
```

| Recipe | Output |
|---|---|
| `00_train_tokenizer` | local tokenizer |
| `01_pretrain_tiny_gpt` | base GPT checkpoint |
| `02_sft_tiny_instruct` | instruction-tuned checkpoint |
| `03_preference_tiny` | preference-tuned checkpoint |
| `04_grpo_tiny_math` | verifier-reward checkpoint |
| `05_eval_all` | consolidated evaluation results |

The defaults favor quick validation. Common overrides include:

```bash
MAX_STEPS=3000 PRESET=gpt-25m bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
ALGORITHM=simpo bash recipes/local_training/03_preference_tiny/run.sh
NUM_GENERATIONS=8 bash recipes/local_training/04_grpo_tiny_math/run.sh
```

Estimate memory before a larger run:

```bash
python scripts/estimate_vram.py --model gpt-25m --method grpo --seq-len 512 --batch-size 1 --num-generations 4
```
