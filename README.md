# minilab

Minilab is my small-scale language-model training lab. The premise is that
the full pretraining to alignment loop, pretraining followed by SFT,
preference optimization, and RLVR, should fit on a single consumer GPU.
On an 8GB laptop I can run every stage end to end in minutes, watch each
checkpoint change the next one, and form intuition about what each step
actually contributes to the model.

That framing matters because most post-training behavior is easier to
reason about at this scale than at production scale. A 10M parameter base
exposes the bias-variance tradeoffs of DPO, the reward-density failure
mode of GRPO, and the formatting-before-content asymmetry of SFT in
single-digit minutes per stage. Production frameworks like TRL, torchtune,
Axolotl, and Megatron solve a different problem: throughput and scale.
Minilab solves visibility.

The main path is:

```text
tokenizer -> tiny GPT pretraining -> SFT -> DPO/SimPO -> GRPO with GSM8K verifier -> eval
```

Every stage lives under `recipes/local_training/` and runs as a single
`bash run.sh`. The diffusion branch (recipes `06` through `09`) follows
the same arc with MDLM in place of GPT, so the two families can be
compared step for step rather than only at the endpoints.

## Cost on my laptop

The reference machine is an NVIDIA RTX 5060 Laptop with 8GB of VRAM. The
main tested configuration is `gpt-10m` with a 4k vocabulary BPE tokenizer,
FP32 training under AdamW, and gradient checkpointing wherever a recipe
turns it on. Wall times below include the first `torch.compile` cold start,
which adds roughly 30 to 60 seconds, and the first HF dataset download.
Both cache between runs, so the first invocation of any stage is the slow
one. Peak VRAM is also written into `run_metrics.json` as
`max_memory_reserved_gb`, which is what to trust over the table when
sizing a longer experiment.

| Stage | Steps | Peak VRAM | Wall time | Result |
| --- | ---: | ---: | ---: | --- |
| 00 tokenizer | - | CPU | ~30s | tokenizer saved, sample sentence roundtrips |
| 01 pretrain `gpt-10m` | 1000 | ~1.5 GB | ~3 min | loss curve looks right; samples have TinyStories cadence but aren't stories yet |
| 02 SFT (Alpaca) | 500 | ~1.2 GB | ~2 min | output shifts from story drift to Q/A shape (content still weak) |
| 03 DPO (HH-RLHF) | 300 | ~1.3 GB | ~2 min | chosen margin stays positive on most pairs |
| 04 GRPO (GSM8K) | 100 | ~1.2 GB | ~12 min | the rollout loop is the wall-time cost; GSM8K accuracy is single-digit and noisy |
| 05 eval | - | ~0.8 GB | ~3 min | per-stage perplexity, Distinct-N, and five sampled completions |
| 06 pretrain `mdlm-25m` | 1000 | ~2.0 GB | ~5 min | denoising loss trends down; samples are token-shaped but not coherent |
| 07 diffusion SFT | 500 | ~1.7 GB | ~3 min | response-token loss drops; ceiling is whatever recipe 06 produced |
| 08 diffusion DPO | 300 | ~2.5 GB | ~6 min | trainable plus frozen reference both fit on 8GB; preference loss stays finite |
| 09 diffusion GRPO | 100 | ~2.3 GB | ~30 min | 64 reverse-diffusion forwards x 2 generations x 100 outer steps is where the time goes |

Peak VRAM stays under 3 GB across the defaults, which leaves comfortable
headroom to scale up to `PRESET=gpt-25m`, a larger `BATCH_SIZE`, or longer
`MAX_NEW_TOKENS` before the 8GB ceiling becomes a problem. The natural
workflow is to call `scripts/estimate_vram.py` first so that any of these
knobs that would overshoot is caught before the run starts rather than
several minutes in.

## What the defaults can and cannot show

The defaults are sized so that the entire loop runs in a coffee break, and
that choice has consequences worth stating up front. The interesting object
to study at this scale is the loss curve and the qualitative shift between
checkpoints, not absolute task performance. Story-level coherence on
TinyStories emerges around `gpt-25m` trained for roughly 3000 steps. Below
that threshold the model has the unigram and short-range statistics but
not the longer-range templates, and reading the samples gives a misleading
impression of what training has done.

Tiny models pick up formatting much faster than content. Both SFT and
DPO or SimPO move response shape, meaning Q and A scaffolding, refusal
patterns, opening style, well before they move task accuracy. Shape lives
in the final softmax distribution where a few thousand examples are
enough to re-weight common tokens. Accuracy lives in representations the
base has not yet learned, and re-weighting cannot conjure them. This is
the dominant aesthetic shift between recipes 01 and 03.

GRPO and RLVR are non-bootstrapping in a strict sense. If the SFT base
produces zero answer-shaped completions, every rollout scores zero, the
group-relative advantage is zero, and the gradient is zero. RL cannot
teach a behavior the base assigns near-zero probability to in the first
place; the practical fix is to train recipe 02 longer rather than to
push GRPO hyperparameters.

The diffusion branch is the less-validated track. Recipes complete and
metrics behave, but at matched parameters and matched compute, masked
diffusion language models reach a given coherence at roughly an order of
magnitude more samples than an autoregressive model. Each token is
supervised through a stochastic timestep expectation rather than a
deterministic next-step target, so the same gradient budget carries
strictly less information per token. The bias-variance balance shifts
toward variance and convergence is correspondingly slower.

The main tested family is GPT-style tiny models. Mamba, Hymba, xLSTM,
ByteLatent, and the diffusion variants are wired into the registries
mainly so the comparison scripts have something honest to compare
against. They are not paved paths through alignment.

HF import currently accepts only Llama-compatible weights. Qwen3 and
Gemma3 round-trip cleanly through inspection and generation, but
`_native_config` in `scripts/import_hf.py` rejects anything whose
`model_type` is not `llama`. Wiring them up is mostly mapping work
rather than capability work: Qwen3 needs the attention-bias and untied
embedding paths, and Gemma3 needs local-global RoPE and per-layer
embedding dimensions. Both surface in the native GPT config; only the
importer is missing the bridge.

## Install

Requires Python 3.10 or newer.

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[data]"     # dataset-backed scripts
python -m pip install -e ".[logging]"  # aim logging
python -m pip install -e ".[dev]"      # pytest and ruff
```

## Local training recipes

The main path lives under `recipes/local_training/`. The recipes pull
TinyStories, Alpaca, HH-RLHF or UltraFeedback, and GSM8K through Hugging
Face Datasets, so the data extra needs to be installed first:

```bash
python -m pip install -e ".[data]"
```

Autoregressive run order:

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
bash recipes/local_training/02_sft_tiny_instruct/run.sh
bash recipes/local_training/03_preference_tiny/run.sh
bash recipes/local_training/04_grpo_tiny_math/run.sh
bash recipes/local_training/05_eval_all/run.sh
```

Diffusion run order (uses the same tokenizer):

```bash
bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
bash recipes/local_training/07_sft_tiny_diffusion/run.sh
bash recipes/local_training/08_preference_tiny_diffusion/run.sh
bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

The diffusion branch mirrors the autoregressive branch stage for stage, but
keeps the diffusion objective at every stage rather than collapsing the
model into a next-token predictor for alignment:

| Stage | AR path | Diffusion path |
| --- | --- | --- |
| Pretraining | next-token prediction | masked denoising |
| SFT | response-only label loss | prompt-fixed response denoising |
| Preference tuning | DPO/IPO/SimPO/etc. over response log-probs | DPO/VRPO over diffusion loss proxies |
| RLVR | GRPO/RLOO/etc. with generated completions | GRPO over reverse denoising trajectories |

Each recipe ships a `README.md`, `config.yaml`, `run.sh`,
`expected_metrics.md`, and `sample_outputs.md`. Environment overrides
like `PRESET=gpt-25m`, `MAX_STEPS=3000`, or `ALGORITHM=simpo` are the
intended way to scale a stage up without touching the script.

## Tiny presets

A preset is a pre-sized model configuration chosen to fit a common laptop
memory budget. The point of having a handful of these rather than asking
the user to set every dimension is that it makes cross-family comparisons
honest: `gpt-25m`, `mamba-25m`, and `mdlm-25m` are roughly matched in
parameter count, so a behavioral difference between them is closer to a
difference in inductive bias than in capacity.

| Preset | Family | Default context | Approx params | Use case |
| --- | --- | ---: | ---: | --- |
| `gpt-10m` | GPT | 512 | ~7M-10M | default tiny training path |
| `gpt-25m` | GPT | 512 | ~20M-25M | larger SFT/preference runs |
| `gpt-60m` | GPT | 1024 | ~52M-59M | stretch local GPT runs |
| `mamba-25m` | Mamba | 512 | ~22M-29M | SSM comparison runs |
| `mdlm-25m` | MDLM | 512 | ~26M-31M | diffusion LM experiments |

The parameter count moves with the tokenizer vocabulary because the
embedding matrix dominates non-attention parameters at this scale. The
ranges above cover the recipe default 4k vocabulary through a 16k
vocabulary.

Use presets directly:

```bash
python scripts/pretrain_lm.py --tokenizer tokenizer.json --preset gpt-10m
python scripts/sft.py --tokenizer tokenizer.json --preset gpt-10m
python scripts/pretrain_diffusion.py --tokenizer tokenizer.json --preset mdlm-25m
```

Estimate memory before a run:

```bash
python scripts/estimate_vram.py \
  --model gpt-25m \
  --method grpo \
  --seq-len 512 \
  --batch-size 1 \
  --num-generations 4
```

Every training run writes PyTorch allocator measurements into
`run_metrics.json`, both in the final checkpoint directory and in the
recipe save directory. On CUDA that file records `max_memory_allocated_gb`
and `max_memory_reserved_gb` from `torch.cuda` peak memory statistics,
which together are a more reliable picture of what the run actually used
than the estimator's a-priori guess.

## Hugging Face bridge

The HF bridge handles curated sub-1B causal LMs. The intent is narrow and
worth being explicit about: inspect them, generate from them, or import a
compatible checkpoint into the native Minilab format so that it goes
through the same trainers as everything else. It is not a general HF
loader, and it does not aim to be. Today only Llama-compatible weights
import cleanly. SmolLM2 works end to end; Qwen3 and Gemma3 round-trip
through inspection and generation but the import path rejects them
until their weight mappings have been validated against the native GPT
config.

```bash
python -m pip install -e ".[data,hf]"
python scripts/hf_inspect.py --list-presets
python scripts/hf_generate.py --model smollm2-135m-instruct --device cuda
bash recipes/hf_to_native/02_import/run.sh
bash recipes/hf_to_native/03_sft_imported/run.sh
bash recipes/hf_to_native/04_preference_imported/run.sh
bash recipes/hf_to_native/05_grpo_imported/run.sh
```

Curated aliases include `smollm2-135m`, `smollm2-360m`, `gemma3-270m`,
and `qwen3-0.6b`, with instruct or base variants where the upstream
release provides them. The full list comes from
`scripts/hf_inspect.py --list-presets`, and the recipes that drive the
import-and-train flow live under `recipes/hf_to_native/`.

## Post-training transfer lab

Minilab also exposes the mechanism-transfer lane from `rl-experiments` under
the same local LM loop. `scripts/grpo.py` can run GRPO-family baselines,
candidate-target methods, influence-allocation methods, replay/freshness
probes, reward-uncertainty variants, and VPO on GSM8K or tiny verifier tasks,
with optional JSONL metrics, trajectory traces, local staleness sweeps, and
tiny code-repair/tool-call verifier tasks:

```bash
python scripts/grpo.py \
  --algorithm tpo \
  --task format_answer \
  --tokenizer tokenizer.json \
  --checkpoint checkpoints/sft/step_500 \
  --rl-metrics-every 1 \
  --rl-trace-samples 8
```

The design remains laptop-scale: local rollouts, deterministic verifiers,
component rewards, failure galleries, and memory metrics rather than a
production rollout cluster. See `docs/post_training_transfer_lab.md`.
Planned sweeps and result slots live in `docs/research_questions.md`.

## Running scripts directly

The recipe wrappers are convenience, not necessity. The underlying
scripts accept the same flags, so anything a recipe does can be driven
from the command line directly:

```bash
python scripts/train_tokenizer.py --save tokenizer.json
python scripts/pretrain_lm.py --tokenizer tokenizer.json
python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
```

```bash
python scripts/sft.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
python scripts/preference.py --algorithm dpo --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
```

```bash
python scripts/pretrain_diffusion.py --tokenizer tokenizer.json --model mdlm
python scripts/sample_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
```

```bash
python scripts/sft_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
python scripts/dpo_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion_sft/step_3000
python scripts/grpo_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion_sft/step_3000
```

## What else is in here

Beyond the main path the registries carry a set of alternative
implementations. These exist primarily so the comparison scripts have
real, equivalently-sized baselines to compare against rather than
strawmen.

- Other LM families: Mamba and Mamba-2, Hybrid, Hymba, xLSTM, ByteLatent.
- Attention ablations: MHA/MQA/GQA, sparse/local/block routes, MLA,
  compressed attention, and a scoped training-time Lighthouse reference.
- Diffusion LMs: MDLM, SEDD, D3PM, and block diffusion.
- Preference-optimization variants: IPO, CPO, ORPO, RePO, and KTO
  alongside DPO and SimPO.
- Online RL variants: RLOO, GSPO, DAPO, and PPO alongside GRPO.
- Tokenizer variants: BPE, WordPiece, Unigram, character, and byte.

The natural entry points for these are the comparison scripts
(`scripts/compare_attention.py`, `scripts/compare_position.py`,
`scripts/compare_connection.py`, and `scripts/compare_diffusion.py`).
The full alignment pipeline has only been driven end to end on GPT-style
models, so for the alternatives the recommendation is to use them as
controlled ablations rather than as production paths through alignment.

## Package contents

The package is organized around registries. Models, tokenizers, attention
layers, position encodings, feed-forward layers, trainers, schedulers,
samplers, and tasks are all selected by string name, which keeps the
scripts thin and makes swapping a component for a comparison run a
one-line change.

Package areas:

- `minilab/tokenizers/`: BPE, WordPiece, Unigram, character, and byte tokenizers.
- `minilab/nn/`: attention, position encodings, normalization, feed-forward
  layers, MoE layers, residual connections, SSM blocks, diffusion blocks, and
  optimizers.
- `minilab/models/`: GPT, Mamba, Mamba-2, Hybrid, Hymba, xLSTM, ByteLatent,
  MDLM, SEDD, D3PM, and block diffusion models.
- `minilab/trainer.py`, `minilab/alignment.py`,
  `minilab/preference_alignment.py`, `minilab/online_rl.py`, and
  `minilab/diffusion_alignment.py`: pretraining, SFT, preference optimization,
  online RL, and diffusion alignment trainers.
- `minilab/data.py` and `minilab/tasks/`: dataset helpers for TinyStories,
  text8, WikiText-103, OpenWebText, Alpaca, Dolly, Anthropic HH, UltraFeedback,
  and GSM8K.
- `minilab/diffusion.py`, `minilab/diffusion_sampling.py`, and
  `minilab/generation.py`: diffusion schedules, diffusion sampling, and
  autoregressive generation.
- `minilab/evaluation.py`, `minilab/evalbench.py`, `minilab/verifiers.py`, and
  `minilab/diagnostics.py`: metrics, benchmark helpers, verifier utilities, and
  diagnostics.
- `minilab/base.py`, `minilab/config.py`, `minilab/checks.py`,
  `minilab/losses.py`, and `minilab/registry.py`: shared infrastructure.

## Scripts

- `scripts/train_tokenizer.py`: train a tokenizer implementation.
- `scripts/pretrain_lm.py`: pretrain GPT, Mamba, Mamba-2, Hybrid, Hymba, xLSTM,
  or ByteLatent models.
- `scripts/pretrain_diffusion.py`: pretrain MDLM, SEDD, D3PM, or block diffusion
  models.
- `scripts/generate.py` and `scripts/sample_diffusion.py`: sample from saved
  autoregressive and diffusion checkpoints.
- `scripts/sft.py`, `scripts/preference.py`, and `scripts/grpo.py`: run SFT,
  offline preference optimization, and online RL for autoregressive models.
- `scripts/sft_diffusion.py`, `scripts/dpo_diffusion.py`, and
  `scripts/grpo_diffusion.py`: run diffusion SFT, preference tuning, and GRPO.
- `scripts/hf_inspect.py` and `scripts/hf_generate.py`: inspect and sample
  from curated sub-1B Hugging Face causal LMs.
- `scripts/import_hf.py`: import compatible Hugging Face causal LMs into
  Minilab's native checkpoint/tokenizer format.
- `scripts/evaluate.py` and `scripts/evaluate_text8.py`: evaluate checkpoints.
- `scripts/estimate_vram.py`: estimate rough memory usage before a run.
- `scripts/compare_attention.py`, `scripts/compare_position.py`,
  `scripts/compare_connection.py`, and `scripts/compare_diffusion.py`: run
  comparison experiments.
- `scripts/common.py`: shared helpers for the script entry points.

## Dependencies

Core dependencies are `torch`, `numpy`, `regex`, `tqdm`, and `pyyaml`. Optional
extras add `datasets`, `aim`, `flash-attn`, `pytest`, and `ruff`.
