# Minilab

Minilab is a laptop-GPU-friendly lab for training and post-training compact GPT language models. The project keeps one model family and exposes research variation through reusable Transformer components rather than separate architecture stacks.

## What the experiments found

Minilab is also an inspectable testbed for asking when small-model RL with
verifiable rewards is scientifically justified. Its current two-tool study
found:

- Observation-conditioned trajectory SFT plus canonical replay raised held-out
  two-tool success by 68.75 percentage points on average across three matched
  seeds while passing a predeclared canonical-preservation gate
  ([trajectory and replay results](research/README.md#5-observation-conditioned-supervision-supplied-the-missing-state)).
- A subsequent ten-step learned RL continuation improved by 5.21 points over
  matched learning-rate-zero controls, but its paired 95% interval was
  [-6.65, +17.07] points. The experiment therefore rejected a longer
  continuation rather than promoting an unstable point estimate
  ([initial matched RL result](research/README.md#7-the-first-paired-rl-estimate-determined-the-next-measurement)).
- A fresh five-seed competence study then found repeatable 20-step RL efficacy
  at SFT step 50: all five learned-minus-control effects were positive, with a
  +20.31-point mean and paired 95% interval of [+3.07, +37.56] points, while
  every predeclared standard metric was preserved
  ([competence and efficacy result](research/README.md#8-the-competence-map-measured-information-available-under-a-fixed-budget)).
- The nominal competence boundary was not an intrinsic learning threshold.
  Under a longer rollout budget, even the step-25 predecessor improved in all
  five seeds (+6.25 points, 95% CI [+2.62, +9.88]) despite the short selector
  observing no mixed groups. One step-25 seed regressed standard behavior, and
  the difference-in-differences between SFT levels was inconclusive. The safer
  interpretation is a budget-conditioned boundary for *reliable, preserved*
  learning—not proof that SFT step 50 uniquely enables RL
  ([claim calibration](research/README.md#8-the-competence-map-measured-information-available-under-a-fixed-budget)).
- On a second fresh block, the existing objective again beat zero-learning
  controls (+17.19 points, 95% CI [+9.07, +25.30]); all four factorial arms
  improved in all five seeds. But the primary between-objective reward contrast
  remained inconclusive at +5.00 points, 95% CI [-1.35, +11.35]. The combined
  centered/fixed-budget arm did shorten completions by 5.30 tokens, 95% CI
  [-10.24, -0.36], so the identified objective effect was behavioral—not an
  objective winner or capability claim
  ([objective-factorization result](research/README.md#9-objective-factorization-separated-efficacy-from-response-geometry)).
- The useful decomposition was **reachability -> preservation -> efficacy**:
  verifier-guided RL has no task-success direction in all-zero-reward groups,
  and new SFT competence is not acceptable if it erases existing behavior.

The complete experimental record, equations, limitations, and next
discriminating experiments are in
[`research/README.md`](research/README.md).

## What is included

- Character, byte, BPE, unigram, WordPiece, and Hugging Face tokenizers.
- GPT pretraining with configurable attention, position encoding, normalization, FFN/MoE, residual connections, and optimizers.
- Supervised fine-tuning and preference methods including DPO, IPO, CPO, ORPO, RePO, SimPO, and KTO.
- Online reinforcement-learning methods including PPO, GRPO, Dr.GRPO, GSPO, RLOO, DAPO, TPO, VPO, and related research lanes.
- Rule-based and learned verifiers, evaluation utilities, diagnostics, and checkpoint metadata validation.
- Native SmolLM2/Qwen3 Hugging Face import, LoRA, and post-training recipes.

## Install

```bash
python -m pip install -e ".[data,dev]"
```

PyTorch is the only heavyweight required dependency. Optional extras are
available for Hugging Face integration, logging, FlashAttention, and procedural
reasoning tasks.

## Local end-to-end workflow

Each recipe is self-contained and can be run from the repository root:

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
bash recipes/local_training/02_sft_tiny_instruct/run.sh
bash recipes/local_training/03_preference_tiny/run.sh
bash recipes/local_training/04_grpo_tiny_math/run.sh
bash recipes/local_training/05_eval_all/run.sh
```

The defaults are intentionally small. Override recipe environment variables such as `MAX_STEPS`, `BATCH_SIZE`, `SEQ_LEN`, and `PRESET` when you want a longer run.

Available presets:

| Preset | Typical role |
|---|---|
| `gpt-10m` | quick local iteration |
| `gpt-25m` | larger laptop experiments |
| `gpt-60m` | stretch runs with more headroom |

## Direct CLI examples

Train a tokenizer and model:

```bash
python scripts/train_tokenizer.py --dataset tinystories --output tokenizer.json
python scripts/pretrain_lm.py --tokenizer tokenizer.json --preset gpt-10m
```

Fine-tune and run preference optimization:

```bash
python scripts/sft.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
python scripts/preference.py --algorithm dpo --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
```

Run verifier-guided training and generation:

```bash
python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/grpo/step_1000 --prompt "Solve 12 + 30."
```

Inspect or import a Hugging Face model:

```bash
python scripts/hf_inspect.py --model HuggingFaceTB/SmolLM2-135M
python scripts/import_hf.py --model HuggingFaceTB/SmolLM2-135M --output checkpoints/imported/smollm2-135m
```

See `recipes/hf_to_native/` for the complete import, SFT, preference, and GRPO sequence.

Import Qwen3-0.6B and train native Q/V LoRA adapters:

```bash
python scripts/import_hf.py \
  --model qwen3-0.6b \
  --save-dir checkpoints/imported/qwen3-0.6b \
  --max-seq-len 512 \
  --device cpu \
  --verify
python scripts/sft.py \
  --tokenizer checkpoints/imported/qwen3-0.6b/tokenizer.json \
  --checkpoint checkpoints/imported/qwen3-0.6b \
  --lora-rank 8 \
  --lora-alpha 16
```

Run the deterministic two-turn agent smoke:

```bash
python scripts/grpo.py \
  --algorithm agentic_turn \
  --task agentic_calculator \
  --tokenizer checkpoints/local_training/tokenizer.json \
  --checkpoint checkpoints/local_training/sft/step_500 \
  --save-dir checkpoints/agentic_turn \
  --max-steps 10 \
  --batch-size 1 \
  --num-generations 4
```

Teach an imported instruction model the exact raw-code, tool-call, and final
answer envelopes before online RL:

```bash
python scripts/sft.py \
  --dataset structured_output \
  --tokenizer checkpoints/imported/qwen3-0.6b/tokenizer.json \
  --checkpoint checkpoints/imported/qwen3-0.6b \
  --save-dir checkpoints/imported/qwen3-0.6b-structured-sft \
  --lora-rank 4 \
  --lora-alpha 8 \
  --max-steps 50
```

For an efficacy claim, compare learned and learning-rate-zero continuations
from the same source checkpoints on identical held-out prompts. The research
note records the paired analysis and preservation gates used in the completed
study.

## GPT component experiments

The single GPT stack supports controlled experiments without multiplying model families:

- Attention: MHA, MQA, GQA, sparse/local/compressed variants, latent attention, and recurrent attention variants.
- Position: learned, sinusoidal, RoPE, YaRN, ALiBi, T5 relative bias, and related methods.
- FFN: dense gated FFNs and several MoE routing strategies.
- Connections: residual, hyperconnection, manifold hyperconnection, and research variants.
- Training: AdamW, Lion, Muon, soft-Muon, checkpointing, MTP, LayerSkip, and auxiliary objectives.

Comparison scripts under `scripts/compare_*.py` run matched GPT experiments for individual component axes.

## Project layout

- `minilab/models/gpt.py`: the model and its configuration contract.
- `minilab/nn/`: reusable attention, position, normalization, FFN, MoE, connection, and optimizer components.
- `minilab/trainer.py`: pretraining loop, checkpointing, resume validation, and optimizer wiring.
- `minilab/preference_alignment.py` and `minilab/online_rl.py`: post-training algorithms.
- `minilab/data.py`: language-model, SFT, preference, verifier, and evaluation datasets.
- `minilab/generation.py`: autoregressive and verified decoding.
- `scripts/`: runnable training, evaluation, import, diagnostics, and comparison entry points.
- `recipes/`: reproducible local and Hugging Face workflows.

## Verification

```bash
python -m compileall -q minilab scripts
```

Checkpoints and local caches are intentionally excluded from version control.
