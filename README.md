# Minilab

Minilab is a small language-model lab built to run the entire
pretraining-through-RLVR loop on a single laptop GPU, where each stage
finishes in minutes. It is built for inspectability rather than
throughput: one GPT family, with research variation exposed through
interchangeable Transformer components instead of separate architecture
stacks, so post-training behavior can be watched at a scale small enough
to reason about directly.

## What the experiments found

Minilab is also an inspectable testbed for a sharper question than
whether RL helps: when is verifier-guided RL on a small tool-using model
justified at all, rather than a longer run chasing noise. The current
study puts a 0.6B instruction model on an exact two-tool calculator task,
and the shape of the progression is the finding.

The imported policy already carried language and tool priors, yet it
produced no strict successes, so RL had nothing to amplify. What unlocked
the task was not more optimization but supervision that reproduced the
environment transition inside each example: once the second tool call was
demonstrated under the observation that fixes its arguments, held-out
two-tool success rose by roughly 69 points across matched seeds. That
gain initially came at the expense of prior skills, until a
canonical-biased replay mixture turned the handoff into a constrained
curriculum that held them in place.

Only after the task was both reachable and retained did RL earn its place
in the pipeline. An early ten-step continuation was encouraging but
inconclusive, its paired interval spanning [-6.65, +17.07], so rather
than promote an unstable point estimate the study raised resolution
instead of budget. A fresh five-seed, twenty-step block then improved
over matched learning-rate-zero controls in every seed, by 20.31 points
on average (95% CI [3.07, 37.56]), with all monitored standard behavior
preserved. The competence boundary proved narrower than it first looked:
under a longer rollout budget even the weaker predecessor learned, which
places the boundary at finite-budget observability rather than at an
intrinsic capability threshold. A closing objective factorization
reproduced the learning effect in all four arms but moved completion
length more clearly than reward, separating a policy's efficacy from the
geometry of how it expresses success.

The decomposition that held the whole study together is
**reachability -> preservation -> efficacy**: RL has no task-success
direction in all-zero-reward groups, fresh supervision is worthless if it
erases behavior the model already had, and only a matched control turns a
training curve into a claim about learning. The complete record, with
equations, exact seed-level intervals, limitations, and the next
discriminating experiments, is in
[`research/README.md`](research/README.md).

## What is included

- Character, byte, BPE, unigram, WordPiece, and Hugging Face tokenizers.
- GPT pretraining with configurable attention, position encoding, normalization, FFN/MoE, residual connections, and optimizers.
- Supervised fine-tuning and preference methods including DPO, IPO, CPO, ORPO, RePO, SimPO, and KTO.
- Online reinforcement-learning methods including PPO, GRPO, Dr.GRPO, GSPO, RLOO, DAPO, TPO, and VPO.
- Rule-based and learned verifiers, evaluation utilities, diagnostics, and checkpoint metadata validation.
- Native SmolLM2/Qwen3 Hugging Face import, LoRA, and post-training recipes.

## Install

```bash
python -m pip install -e ".[data,dev]"
```

Minilab targets Python 3.10 or newer, and PyTorch is the only heavyweight
required dependency. The core install adds only `numpy`, `regex`, `tqdm`,
and `pyyaml` alongside it. The optional extras stay narrow, and each maps
to a specific job:

- `data` pulls `datasets` for the recipes that read TinyStories, Alpaca,
  HH-RLHF, UltraFeedback, and GSM8K through Hugging Face.
- `hf` adds `transformers`, `huggingface_hub`, and `safetensors` for the
  inspect, generate, and import path in `recipes/hf_to_native/`.
- `flash` (or `triton`) enables the in-repo Triton attention backend.
- `logging` adds Aim logging, `reasoning` adds procedural reasoning
  tasks, and `dev` adds pytest and ruff.

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

The defaults are intentionally small. Override recipe environment
variables such as `MAX_STEPS`, `MAX_EXAMPLES`, `BATCH_SIZE`, `SEQ_LEN`,
and `PRESET` when you want a longer run.

Three GPT presets cover the useful laptop range, differing only in width,
depth, and context length:

| Preset | dim | layers | heads | context | Typical role |
|---|---:|---:|---:|---:|---|
| `gpt-10m` | 256 | 6 | 8 | 512 | tokenizer, pretraining, SFT, and preference runs |
| `gpt-25m` | 384 | 8 | 8 | 512 | larger, more realistic alignment experiments |
| `gpt-60m` | 512 | 12 | 8 | 1024 | stretch runs with more headroom |

Parameter count is not fixed by the preset alone. The embedding and output
projection dominate at this scale, so the total moves with the tokenizer
vocabulary: at the default 4k vocabulary `gpt-10m` is about 7.5M
parameters, and a 16k vocabulary lifts the same preset toward roughly 13M,
almost entirely through the token embeddings.

## What the tiny runs can and cannot show

The defaults are sized so the whole loop runs in a coffee break, and that
sizing decides what the runs can teach. The object worth studying here is
the loss curve and the qualitative shift from one checkpoint to the next,
not absolute task scores. Three regularities show up cleanly at this
scale, and they are most of the reason the lab exists.

Story-level coherence on TinyStories arrives around `gpt-25m` trained for
roughly 3000 steps. Below that the model has the unigram and short-range
statistics but not the longer-range templates, so `gpt-10m` samples read
as fluent but narratively flat and give a misleading impression of what
training has done.

Formatting moves faster than content. SFT and the preference methods
shift response shape, the question-and-answer scaffolding and opening
style, well before they move task accuracy. Shape lives in the final
softmax, where a few thousand examples re-weight common tokens, while
accuracy would need representations the base does not yet have. This
asymmetry is the dominant change between recipes 01 and 03, and it recurs
all the way up the alignment stack.

Verifier-guided RL does not bootstrap. If the SFT base produces no
answer-shaped completions, every rollout scores zero, the group-relative
advantage is zero, and so is the gradient: RL cannot lift a behavior the
base assigns almost no probability to in the first place. When GRPO
stalls, the fix is upstream, in training recipe 02 longer, not in RL
hyperparameters. The research study in
[`research/README.md`](research/README.md) is the same lesson at larger
scale, where supervised state coverage rather than more optimization was
what made the task reachable at all.

## Cost and memory

The runs are cheap enough to iterate on, and the repo measures its own
cost rather than asking you to trust a table. `scripts/estimate_vram.py`
gives an a-priori estimate before a run, and every training run then
writes `run_metrics.json` into both the final checkpoint directory and
the recipe save root. On CUDA that file records `max_memory_allocated_gb`
and `max_memory_reserved_gb` from PyTorch's peak-memory statistics, which
are the numbers to trust when sizing a longer experiment. Wall time is
dominated by the rollout loop in recipe 04: sampling a group of
completions for every prompt costs far more per step than the supervised
stages, which is why its defaults are the most conservative in the track.

```bash
python scripts/estimate_vram.py --model gpt-25m --method grpo --seq-len 512 --batch-size 1 --num-generations 4
```

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

## Post-training RL methods

Recipe 04 runs plain GRPO, but `scripts/grpo.py` is the entry point for a
wider menu of online-RL methods that share one rollout-score-update loop:
GRPO, Dr.GRPO, DAPO, GSPO, RLOO, PPO, and the more experimental TPO and
VPO, each selected with `--algorithm`. Collecting them behind a single
script turns them into a controlled comparison, since the same verifier
and the same tiny policy can be pointed at each in turn to see where they
actually diverge. Dr.GRPO, for instance, drops GRPO's
group-standard-deviation scaling and response-length loss normalization
in favor of centered rewards and a fixed generation-budget token
denominator, so its difference from GRPO is a change in the geometry of
the update rather than in the reward.

The reward does not have to come from GSM8K. Four deterministic-verifier
tasks, `format_answer`, `mini_arithmetic`, `tool_call_json`, and
`tiny_code_repair`, each return a clean pass or fail with no dataset at
all, which isolates an algorithm's behavior from dataset noise.

```bash
python scripts/grpo.py \
  --algorithm tpo \
  --task format_answer \
  --tokenizer tokenizer.json \
  --checkpoint checkpoints/sft/step_500 \
  --rl-metrics-every 1 \
  --rl-trace-samples 8
```

The trainer writes per-step metrics to `online_rl_metrics.jsonl` and, when
`--rl-trace-samples` is set, sampled rollout trajectories to
`trajectories.jsonl`, so a run that goes wrong can be read directly from
its traces rather than guessed at from the loss curve.

## GPT component experiments

The single GPT stack supports controlled experiments without multiplying model families:

- Attention: MHA, MQA, GQA, sparse/local/compressed variants, latent attention, and recurrent attention variants.
- Position: learned, sinusoidal, RoPE, YaRN, ALiBi, T5 relative bias, and related methods.
- FFN: dense gated FFNs and several MoE routing strategies.
- Connections: residual, hyperconnection, manifold hyperconnection, and research variants.
- Training: AdamW, Lion, Muon, soft-Muon, checkpointing, MTP, LayerSkip, and auxiliary objectives.

Comparison scripts under `scripts/compare_*.py` run matched GPT experiments for individual component axes.

## Package contents

The package is organized around registries: models, tokenizers, attention
layers, position encodings, feed-forward layers, optimizers, trainers, and
tasks are all selected by string name, which keeps the scripts thin and
makes swapping one component for a comparison run a one-line change.

- `minilab/models/`: the GPT model and its configuration contract in
  `gpt.py`, with shared building blocks in `transformer_utils.py`.
- `minilab/nn/`: the interchangeable components, one concern per module,
  covering attention (standard, sparse, compressed, latent, linear, and a
  Triton backend), position encodings, normalization, feed-forward and MoE
  layers, residual connections, optimizers, and LoRA adapters.
- `minilab/trainer.py`: the pretraining loop, checkpointing, resume
  validation, optimizer wiring, and the `run_metrics.json` writer.
- `minilab/alignment.py`, `minilab/preference_alignment.py`, and
  `minilab/online_rl.py`: the SFT, offline-preference, and online-RL
  trainers, with shared plumbing in `minilab/alignment_common.py` and
  agentic rollout lanes in `minilab/agentic_lanes.py`.
- `minilab/data.py` and `minilab/tasks/`: dataset helpers for TinyStories,
  Alpaca, HH-RLHF, UltraFeedback, and GSM8K, plus the agentic calculator,
  the deterministic verifier toys, and reasoning-gym tasks.
- `minilab/generation.py`: autoregressive and verifier-guided decoding.
- `minilab/evaluation.py`, `minilab/evalbench.py`, `minilab/verifiers.py`,
  `minilab/diagnostics.py`, and `minilab/rl_diagnostics.py`: metrics,
  benchmark helpers, verifier utilities, and training diagnostics.
- `minilab/hf_presets.py` and `minilab/hf_cache.py`: the curated Hugging
  Face alias table and the in-workspace download cache.
- `minilab/base.py`, `minilab/config.py`, `minilab/presets.py`,
  `minilab/checks.py`, `minilab/losses.py`, and `minilab/registry.py`:
  shared infrastructure and the model-preset definitions.

## Scripts

- `scripts/train_tokenizer.py`: train a tokenizer.
- `scripts/pretrain_lm.py`: pretrain a GPT model.
- `scripts/generate.py`: sample from a saved checkpoint.
- `scripts/sft.py`, `scripts/preference.py`, and `scripts/grpo.py`: run
  SFT, offline preference optimization, and online RL.
- `scripts/evaluate.py` and `scripts/evaluate_text8.py`: evaluate
  checkpoints on validation perplexity, generation diversity, and text8
  bits per character.
- `scripts/estimate_vram.py`: estimate memory before a run.
- `scripts/hf_inspect.py`, `scripts/hf_generate.py`, and
  `scripts/import_hf.py`: inspect, sample from, and import curated sub-1B
  Hugging Face causal language models.
- `scripts/compare_attention.py`, `scripts/compare_position.py`,
  `scripts/compare_connection.py`, and `scripts/compare_mtp.py`: matched
  component-ablation experiments.
- `scripts/benchmark_generation.py` and
  `scripts/benchmark_msa_attention.py`: generation and attention-kernel
  benchmarks.
- `scripts/common.py` and `scripts/hf_common.py`: shared helpers for the
  entry points.
- `recipes/`: the reproducible local-training and Hugging-Face workflows
  that drive these scripts end to end.

## Verification

```bash
python -m compileall -q minilab scripts
```

Checkpoints and local caches are intentionally excluded from version control.
