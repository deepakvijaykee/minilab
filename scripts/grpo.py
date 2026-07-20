"""GRPO on GSM8K math problems.

    python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
"""

import argparse
import math
import torch
from common import (
    MODEL_CHOICES,
    load_model_checkpoint,
    reject_supplied,
    require_checkpoint_path,
    resolve_default,
    resolve_save_every,
)
from minilab.tokenizers import load_tokenizer
from minilab.data import load_gsm8k
from minilab.alignment import (
    AgenticTrajectoryTrainer,
    AgenticTrajectoryTrainConfig,
    AgenticTurnTrainConfig,
    AgenticTurnTrainer,
    ASPOTrainer,
    DAPOTrainConfig,
    DAPOTrainer,
    DGTrainer,
    DrGRPOTrainConfig,
    DrGRPOTrainer,
    FilteredDGTrainer,
    FreshDGTrainer,
    GRPOTrainConfig,
    GRPOTrainer,
    GRPOLiteTrainConfig,
    GRPOLiteTrainer,
    GroupPGTrainConfig,
    GroupPGTrainer,
    GSPOTrainConfig,
    GSPOTrainer,
    KondoTrainer,
    PPOTrainConfig,
    PPOTrainer,
    R2VPOTrainer,
    RLOOTrainConfig,
    RLOOTrainer,
    ReplayDGTrainer,
    RewardVarianceDGTrainer,
    SequenceDGTrainConfig,
    TPONoAnchorTrainer,
    TPOTrainConfig,
    TPOTrainer,
    UncertaintyDGTrainer,
    VPOTrainConfig,
    VPOTrainer,
    online_rl_uses_reference,
    resolve_reference_path,
)
from minilab.online_rl import calculator_agent_turn, calculator_trajectory_turn
from minilab.checks import require
from minilab.diagnostics import optimizer_state_bytes
from minilab.nn.optimizers import DEFAULT_SOFT_MUON_POWER
from minilab.trainer import run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer
from minilab.tasks.gsm8k import batch_reward, extract_answer, reward as gsm8k_reward
from minilab.tasks.agentic_calculator import (
    AGENTIC_CALCULATOR_MIXTURE_CONTRACT,
    make_agentic_calculator_dataset,
    make_agentic_calculator_mixture_dataset,
    make_calculator_trajectory_dataset,
)
from minilab.tasks.reasoning_gym import (
    make_reasoning_gym_dataset,
    parse_reasoning_gym_config,
    reasoning_gym_batch_reward,
)
from minilab.tasks.verifier_toys import (
    format_answer_batch_reward,
    make_format_answer_dataset,
    make_mini_arithmetic_dataset,
    make_tiny_code_repair_dataset,
    make_tool_call_dataset,
    make_verifier_hacking_code_repair_dataset,
    mini_arithmetic_batch_reward,
    tiny_code_repair_batch_reward,
    tool_call_batch_reward,
    verifier_hacking_code_repair_batch_reward,
)
from minilab.generation import generate


p = argparse.ArgumentParser()
p.add_argument(
    "--algorithm",
    choices=[
        "ppo", "grpo", "drgrpo", "grpo_lite", "dapo", "gspo", "rloo",
        "agentic_turn", "agentic_trajectory",
        "tpo", "tpo_no_anchor", "group_pg",
        "vpo",
        "dg", "kondo", "uncertainty_dg", "filtered_dg", "reward_variance_dg",
        "aspo", "r2vpo", "replay_dg", "fresh_dg",
    ],
    default="grpo",
)
p.add_argument(
    "--task",
    choices=[
        "gsm8k", "reasoning_gym", "format_answer", "mini_arithmetic",
        "tool_call_json", "agentic_calculator", "calculator_trajectory",
        "tiny_code_repair",
        "verifier_hacking_code_repair",
    ],
    default="gsm8k",
)
p.add_argument("--reasoning-task", default=None, help="Reasoning Gym generator name; defaults to leg_counting")
p.add_argument("--reasoning-config", default=None, help="JSON object of task-specific training generator options")
p.add_argument("--reasoning-eval-config", default=None, help="JSON object for held-out/OOD generation; defaults to training config")
p.add_argument("--reasoning-train-seed", type=int, default=None, help="defaults to --seed")
p.add_argument("--reasoning-eval-seed", type=int, default=None, help="defaults to --reasoning-train-seed + 1")
p.add_argument(
    "--agentic-suite",
    choices=["canonical", "mixed"],
    default=None,
    help="one-tool agentic calculator training suite; defaults to canonical",
)
p.add_argument(
    "--agentic-start",
    type=int,
    default=None,
    help="first deterministic one- or two-tool stream index; defaults to 0",
)
p.add_argument(
    "--agentic-advantage-estimator",
    choices=["group_standardized", "group_centered"],
    default=None,
    help="three-stage credit estimator; defaults to group_standardized",
)
p.add_argument(
    "--agentic-loss-normalizer",
    choices=["response_length", "generation_budget"],
    default=None,
    help="three-stage token-loss denominator; defaults to response_length",
)
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="override checkpoint model family")
p.add_argument("--save-dir", default="")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=500)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--lora-rank", type=int, default=None, help="enable Q/V LoRA on --checkpoint")
p.add_argument("--lora-alpha", type=float, default=None, help="LoRA scale; defaults to --lora-rank")
p.add_argument("--muon-lr", type=float, default=None, help="defaults to 0.02 for Muon-family optimizers")
p.add_argument("--soft-muon-power", type=float, default=None, help="fixed p=0.4 profile for --optimizer soft_muon")
p.add_argument("--optimizer", choices=["adamw", "lion", "muon", "soft_muon"], default="adamw")
p.add_argument("--num-generations", type=int, default=None, help="defaults to 4 for group policy algorithms")
p.add_argument("--inner-epochs", type=int, default=None, help="inner update epochs per rollout; defaults to 1 for RLOO/GRPO-lite/agentic algorithms and 4 otherwise")
p.add_argument(
    "--kl-coef",
    type=float,
    default=None,
    help="defaults to 0.1 for PPO/GRPO/DrGRPO/GSPO/RLOO/R2VPO; DAPO/GRPO-lite/agentic algorithms accept only 0",
)
p.add_argument(
    "--clip-ratio",
    type=float,
    default=None,
    help="defaults to 0.2 for PPO/GRPO/DrGRPO/VPO and 4e-4 for GSPO; other methods reject it",
)
p.add_argument("--value-clip", type=float, default=None, help="defaults to 0.2 for PPO")
p.add_argument("--value-coef", type=float, default=None, help="defaults to 0.5 for PPO")
p.add_argument("--entropy-coef", type=float, default=None, help="defaults to 0.0 for PPO")
p.add_argument("--gae-lambda", type=float, default=None, help="defaults to 0.95 for PPO")
p.add_argument("--clip-ratio-low", type=float, default=None, help="defaults to 0.2 for DAPO")
p.add_argument("--clip-ratio-high", type=float, default=None, help="defaults to 0.28 for DAPO")
p.add_argument("--safe-length", type=int, default=None, help="defaults to 0 for DAPO")
p.add_argument("--length-penalty", type=float, default=None, help="defaults to 0.0 for DAPO")
p.add_argument("--max-resample", type=int, default=None, help="defaults to 5 for DAPO")
p.add_argument("--tpo-eta", type=float, default=None)
p.add_argument("--vpo-num-candidates", type=int, default=None)
p.add_argument("--vpo-num-scalarizations", type=int, default=None)
p.add_argument("--vpo-dirichlet-alpha", type=float, default=None)
p.add_argument("--dg-eta", type=float, default=None)
p.add_argument("--dg-keep-ratio", type=float, default=None)
p.add_argument("--dg-uncertainty-threshold", type=float, default=None)
p.add_argument("--replay-capacity", type=int, default=None)
p.add_argument("--replay-min-age", type=int, default=None)
p.add_argument("--replay-age-decay", type=float, default=None)
p.add_argument("--staleness-delay", type=int, default=None)
p.add_argument("--drop-stale-after", type=int, default=None)
p.add_argument("--vram-budget-gb", type=float, default=8.0)
p.add_argument("--rl-metrics-every", type=int, default=1)
p.add_argument("--rl-trace-samples", type=int, default=0)
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--max-examples", type=int, default=2000)
p.add_argument("--eval-examples", type=int, default=0, help="0 = full GSM8K test split; synthetic verifier tasks use 50 examples when unset")
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
set_seed(args.seed)

if args.resume_from:
    reject_supplied(args, ("lora_rank", "lora_alpha"), "is restored from --resume-from")
if args.lora_rank is not None:
    require(args.lora_rank > 0, "--lora-rank must be > 0 when supplied")
if args.lora_alpha is not None:
    require(args.lora_rank is not None, "--lora-alpha requires --lora-rank")
    require(math.isfinite(args.lora_alpha) and args.lora_alpha > 0, (
        "--lora-alpha must be finite and > 0 when supplied"
    ))

agentic_algorithm_tasks = {
    "agentic_turn": "agentic_calculator",
    "agentic_trajectory": "calculator_trajectory",
}
if args.algorithm in agentic_algorithm_tasks:
    required_task = agentic_algorithm_tasks[args.algorithm]
    require(args.task == required_task, (
        f"--algorithm {args.algorithm} requires --task {required_task}"
    ))
else:
    require(args.task not in set(agentic_algorithm_tasks.values()), (
        "agentic calculator tasks require their matching agentic algorithm"
    ))

if args.task not in set(agentic_algorithm_tasks.values()):
    reject_supplied(args, ("agentic_suite",), (
        "only applies to --task agentic_calculator"
    ))
    reject_supplied(args, ("agentic_start",), "only applies to agentic tasks")
    agentic_suite = None
    agentic_start = None
else:
    agentic_start = 0 if args.agentic_start is None else args.agentic_start
    require(agentic_start >= 0, "--agentic-start must be >= 0")
    if args.task == "agentic_calculator":
        agentic_suite = args.agentic_suite or "canonical"
    else:
        reject_supplied(args, ("agentic_suite",), (
            "only applies to --task agentic_calculator"
        ))
        agentic_suite = None

trajectory_factor_fields = (
    "agentic_advantage_estimator",
    "agentic_loss_normalizer",
)
if args.algorithm != "agentic_trajectory":
    reject_supplied(
        args,
        trajectory_factor_fields,
        "only applies to --algorithm agentic_trajectory",
    )
    agentic_advantage_estimator = None
    agentic_loss_normalizer = None
else:
    agentic_advantage_estimator = (
        args.agentic_advantage_estimator or "group_standardized"
    )
    agentic_loss_normalizer = (
        args.agentic_loss_normalizer or "response_length"
    )

reasoning_fields = (
    "reasoning_task", "reasoning_config", "reasoning_eval_config",
    "reasoning_train_seed", "reasoning_eval_seed",
)
if args.task != "reasoning_gym":
    reject_supplied(args, reasoning_fields, "only applies to --task reasoning_gym")
else:
    reasoning_task = args.reasoning_task or "leg_counting"
    reasoning_config = parse_reasoning_gym_config(args.reasoning_config, "--reasoning-config")
    reasoning_eval_config = (
        reasoning_config
        if args.reasoning_eval_config is None
        else parse_reasoning_gym_config(args.reasoning_eval_config, "--reasoning-eval-config")
    )
    reasoning_train_seed = args.seed if args.reasoning_train_seed is None else args.reasoning_train_seed
    reasoning_eval_seed = (
        reasoning_train_seed + 1
        if args.reasoning_eval_seed is None
        else args.reasoning_eval_seed
    )
    require(reasoning_eval_seed != reasoning_train_seed, (
        "Reasoning Gym train and held-out evaluation seeds must differ"
    ))

if args.algorithm != "ppo":
    require(args.value_clip is None, "--value-clip only applies to --algorithm ppo")
    require(args.value_coef is None, "--value-coef only applies to --algorithm ppo")
    require(args.entropy_coef is None, "--entropy-coef only applies to --algorithm ppo")
    require(args.gae_lambda is None, "--gae-lambda only applies to --algorithm ppo")
if args.algorithm == "ppo":
    require(args.num_generations is None, "--num-generations only applies to group policy algorithms")
    require(args.optimizer == "adamw", "--optimizer only supports adamw for --algorithm ppo")
if args.algorithm not in {"ppo", "grpo", "drgrpo", "gspo", "vpo"}:
    require(args.clip_ratio is None, "--clip-ratio only applies to PPO/GRPO/DrGRPO/GSPO/VPO")
if args.algorithm not in {"ppo", "grpo", "drgrpo", "gspo", "rloo", "r2vpo", "dapo", "grpo_lite", "agentic_turn", "agentic_trajectory"}:
    require(args.kl_coef is None, "--kl-coef only applies to PPO/GRPO/DrGRPO/GSPO/RLOO/R2VPO; DAPO/GRPO-lite/agentic algorithms accept only 0")
if args.algorithm != "dapo":
    require(args.clip_ratio_low is None, "--clip-ratio-low only applies to --algorithm dapo")
    require(args.clip_ratio_high is None, "--clip-ratio-high only applies to --algorithm dapo")
    require(args.safe_length is None, "--safe-length only applies to --algorithm dapo")
    require(args.length_penalty is None, "--length-penalty only applies to --algorithm dapo")
    require(args.max_resample is None, "--max-resample only applies to --algorithm dapo")
if args.algorithm not in {"tpo", "tpo_no_anchor"}:
    reject_supplied(args, ("tpo_eta",), "only applies to --algorithm tpo or tpo_no_anchor")
if args.algorithm != "vpo":
    reject_supplied(args, ("vpo_num_candidates", "vpo_num_scalarizations", "vpo_dirichlet_alpha"), (
        "only applies to --algorithm vpo"
    ))
dg_algorithms = {
    "dg", "kondo", "uncertainty_dg", "filtered_dg", "reward_variance_dg",
    "aspo", "r2vpo", "replay_dg", "fresh_dg",
}
if args.algorithm not in dg_algorithms:
    reject_supplied(args, (
        "dg_eta", "dg_keep_ratio", "dg_uncertainty_threshold",
        "replay_capacity", "replay_min_age", "replay_age_decay",
        "staleness_delay", "drop_stale_after",
    ), "only applies to DG-family algorithms")
else:
    if args.algorithm != "kondo":
        reject_supplied(args, ("dg_keep_ratio",), "only applies to --algorithm kondo")
    if args.algorithm != "filtered_dg":
        reject_supplied(args, ("dg_uncertainty_threshold",), "only applies to --algorithm filtered_dg")
    if args.algorithm not in {"replay_dg", "fresh_dg"}:
        reject_supplied(args, (
            "replay_capacity", "replay_min_age", "replay_age_decay",
            "staleness_delay", "drop_stale_after",
        ), "only applies to --algorithm replay_dg or fresh_dg")
    elif args.algorithm != "fresh_dg":
        reject_supplied(args, ("replay_age_decay",), "only applies to --algorithm fresh_dg")
if args.optimizer not in {"muon", "soft_muon"}:
    require(args.muon_lr is None, "--muon-lr only applies to --optimizer muon or soft_muon")
if args.optimizer != "soft_muon":
    require(args.soft_muon_power is None, "--soft-muon-power only applies to --optimizer soft_muon")
if args.soft_muon_power is not None:
    require(
        args.soft_muon_power == DEFAULT_SOFT_MUON_POWER,
        "--soft-muon-power currently supports the fixed p=0.4 coefficient profile",
    )

num_generations = resolve_default(args.num_generations, 4)
value_clip = resolve_default(args.value_clip, 0.2)
value_coef = resolve_default(args.value_coef, 0.5)
entropy_coef = resolve_default(args.entropy_coef, 0.0)
gae_lambda = resolve_default(args.gae_lambda, 0.95)
clip_ratio_low = resolve_default(args.clip_ratio_low, 0.2)
clip_ratio_high = resolve_default(args.clip_ratio_high, 0.28)
safe_length = resolve_default(args.safe_length, 0)
length_penalty = resolve_default(args.length_penalty, 0.0)
max_resample = resolve_default(args.max_resample, 5)
tpo_eta = resolve_default(args.tpo_eta, 1.0)
vpo_num_candidates = resolve_default(args.vpo_num_candidates, 3)
vpo_num_scalarizations = resolve_default(args.vpo_num_scalarizations, 8)
vpo_dirichlet_alpha = resolve_default(args.vpo_dirichlet_alpha, 1.0)
dg_eta = resolve_default(args.dg_eta, 1.0)
dg_keep_ratio = resolve_default(args.dg_keep_ratio, 0.5)
dg_uncertainty_threshold = resolve_default(args.dg_uncertainty_threshold, 0.5)
replay_min_age = resolve_default(args.replay_min_age, 1)
replay_age_decay = resolve_default(args.replay_age_decay, 0.0)
staleness_delay = resolve_default(args.staleness_delay, 0)
drop_stale_after = resolve_default(args.drop_stale_after, 0)
muon_lr = resolve_default(args.muon_lr, 0.02)
soft_muon_power = resolve_default(args.soft_muon_power, DEFAULT_SOFT_MUON_POWER)
inner_epochs = args.inner_epochs
if inner_epochs is None:
    inner_epochs = 1 if args.algorithm in {"rloo", "grpo_lite", "agentic_turn", "agentic_trajectory"} else 4
kl_coef = args.kl_coef
if kl_coef is None:
    kl_coef = 0.0 if args.algorithm in {"dapo", "grpo_lite", "agentic_turn", "agentic_trajectory"} else 0.1
if args.algorithm == "dapo":
    require(kl_coef == 0, "DAPO removes the KL penalty; set --kl-coef 0 or leave it unset")
    require(args.clip_ratio is None, "DAPO uses --clip-ratio-low/--clip-ratio-high; do not set --clip-ratio")
if args.algorithm == "grpo_lite":
    require(kl_coef == 0, "GRPO-lite is reference-free; set --kl-coef 0 or leave it unset")
    require(inner_epochs == 1, "GRPO-lite is a one-update REINFORCE ablation; set --inner-epochs 1 or leave it unset")
if args.algorithm in agentic_algorithm_tasks:
    require(kl_coef == 0, (
        f"{args.algorithm} is reference-free; set --kl-coef 0 or leave it unset"
    ))
    require(inner_epochs == 1, (
        f"{args.algorithm} uses one on-policy update; set --inner-epochs 1 or leave it unset"
    ))
if args.algorithm == "rloo":
    require(args.clip_ratio is None, "RLOO is an unclipped REINFORCE estimator; do not set --clip-ratio")
clip_ratio_default = 4e-4 if args.algorithm == "gspo" else 0.2
clip_ratio = resolve_default(args.clip_ratio, clip_ratio_default)
require(vpo_num_candidates > 1, "--vpo-num-candidates must be > 1")
require(vpo_num_scalarizations > 0, "--vpo-num-scalarizations must be > 0")
require(vpo_dirichlet_alpha > 0, "--vpo-dirichlet-alpha must be > 0")
require(tpo_eta > 0, "--tpo-eta must be > 0")
require(dg_eta > 0, "--dg-eta must be > 0")
require(0 < dg_keep_ratio <= 1, "--dg-keep-ratio must be in (0, 1]")
require(dg_uncertainty_threshold >= 0, "--dg-uncertainty-threshold must be >= 0")
require(replay_min_age >= 0, "--replay-min-age must be >= 0")
require(replay_age_decay >= 0, "--replay-age-decay must be >= 0")
require(staleness_delay >= 0, "--staleness-delay must be >= 0")
require(drop_stale_after >= 0, "--drop-stale-after must be >= 0")
require(drop_stale_after == 0 or drop_stale_after >= staleness_delay, (
    "--drop-stale-after must be 0 or >= --staleness-delay"
))
if staleness_delay > 0 or drop_stale_after > 0:
    require(args.algorithm in {"replay_dg", "fresh_dg"}, (
        "--staleness-delay/--drop-stale-after only apply to replay_dg or fresh_dg"
    ))

tok = load_tokenizer(args.tokenizer)
require(args.eval_examples >= 0, "--eval-examples must be >= 0")
require(args.task == "gsm8k" or args.max_examples > 0, (
    "--max-examples must be > 0 for non-GSM8K tasks"
))
synthetic_eval_examples = 50 if args.eval_examples == 0 else args.eval_examples

model_path = require_checkpoint_path(args.checkpoint, args.resume_from, "GRPO training")
validate_checkpoint_tokenizer(model_path, tok)
ref_path = None
if online_rl_uses_reference(args.algorithm):
    ref_path = resolve_reference_path(args.checkpoint, args.resume_from, args.algorithm.upper())
    validate_checkpoint_tokenizer(ref_path, tok)
model_name, model = load_model_checkpoint(model_path, args.model)
if args.lora_rank is not None:
    model.enable_lora(args.lora_rank, args.lora_alpha)
print(f"Trainable: {model_path} ({model_name}, {model.num_parameters():,} params)")
if ref_path is not None:
    print(f"Frozen reference: {ref_path}")

if args.task == "gsm8k":
    train_ds = load_gsm8k(tok, args.seq_len, max_examples=args.max_examples, split="train")
    eval_ds = load_gsm8k(tok, args.seq_len, max_examples=args.eval_examples, split="test")
    task_reward = batch_reward
    print(f"GSM8K: train={len(train_ds)} test={len(eval_ds)}")
elif args.task == "reasoning_gym":
    train_ds = make_reasoning_gym_dataset(
        tok,
        args.seq_len,
        reasoning_task,
        count=args.max_examples,
        seed=reasoning_train_seed,
        config=reasoning_config,
    )
    eval_ds = make_reasoning_gym_dataset(
        tok,
        args.seq_len,
        reasoning_task,
        count=synthetic_eval_examples,
        seed=reasoning_eval_seed,
        config=reasoning_eval_config,
    )
    task_reward = reasoning_gym_batch_reward
    print(
        f"Reasoning Gym {reasoning_task}: train={len(train_ds)} seed={reasoning_train_seed} "
        f"test={len(eval_ds)} seed={reasoning_eval_seed}"
    )
elif args.task == "format_answer":
    train_ds = make_format_answer_dataset(tok, args.seq_len, count=args.max_examples)
    eval_ds = make_format_answer_dataset(tok, args.seq_len, count=synthetic_eval_examples)
    task_reward = format_answer_batch_reward
    print(f"format_answer: train={len(train_ds)} test={len(eval_ds)}")
elif args.task == "mini_arithmetic":
    train_ds = make_mini_arithmetic_dataset(tok, args.seq_len, count=args.max_examples)
    eval_ds = make_mini_arithmetic_dataset(tok, args.seq_len, count=synthetic_eval_examples)
    task_reward = mini_arithmetic_batch_reward
    print(f"mini_arithmetic: train={len(train_ds)} test={len(eval_ds)}")
elif args.task == "tool_call_json":
    train_ds = make_tool_call_dataset(tok, args.seq_len, count=args.max_examples)
    eval_ds = make_tool_call_dataset(tok, args.seq_len, count=synthetic_eval_examples)
    task_reward = tool_call_batch_reward
    print(f"tool_call_json: train={len(train_ds)} test={len(eval_ds)}")
elif args.task == "agentic_calculator":
    agentic_dataset = (
        make_agentic_calculator_dataset
        if agentic_suite == "canonical"
        else make_agentic_calculator_mixture_dataset
    )
    train_ds = agentic_dataset(
        tok,
        args.seq_len,
        count=args.max_examples,
        start=agentic_start,
    )
    eval_ds = agentic_dataset(
        tok,
        args.seq_len,
        count=synthetic_eval_examples,
        start=agentic_start + args.max_examples,
    )
    print(
        f"agentic_calculator ({agentic_suite}, start={agentic_start}): "
        f"train={len(train_ds)} test={len(eval_ds)}"
    )
elif args.task == "calculator_trajectory":
    train_ds = make_calculator_trajectory_dataset(
        tok,
        args.seq_len,
        count=args.max_examples,
        start=agentic_start,
    )
    eval_ds = make_calculator_trajectory_dataset(
        tok,
        args.seq_len,
        count=synthetic_eval_examples,
        start=agentic_start + args.max_examples,
    )
    print(
        f"calculator_trajectory (start={agentic_start}): "
        f"train={len(train_ds)} test={len(eval_ds)}"
    )
elif args.task == "tiny_code_repair":
    train_ds = make_tiny_code_repair_dataset(tok, args.seq_len, count=args.max_examples)
    eval_ds = make_tiny_code_repair_dataset(tok, args.seq_len, count=synthetic_eval_examples)
    task_reward = tiny_code_repair_batch_reward
    print(f"tiny_code_repair: train={len(train_ds)} test={len(eval_ds)}")
elif args.task == "verifier_hacking_code_repair":
    train_ds = make_verifier_hacking_code_repair_dataset(
        tok, args.seq_len, count=args.max_examples
    )
    eval_ds = make_verifier_hacking_code_repair_dataset(
        tok, args.seq_len, count=synthetic_eval_examples
    )
    task_reward = verifier_hacking_code_repair_batch_reward
    print(
        "verifier_hacking_code_repair (weak proxy reward): "
        f"train={len(train_ds)} test={len(eval_ds)}"
    )
else:
    raise AssertionError(f"unhandled task choice: {args.task}")


def math_reward(batch, completions, completion_mask):
    return task_reward(tok, train_ds.answers, batch, completions, completion_mask)


def print_rl_budget(model, algorithm, seq_len, batch_size, generations, max_new_tokens, config, budget_gb):
    params = model.num_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ref_params = params if online_rl_uses_reference(algorithm) else 0
    rollout_multiplier = 1 if algorithm == "ppo" else generations
    if algorithm == "vpo":
        rollout_multiplier *= config.vpo_num_candidates
    if algorithm in {"agentic_turn", "agentic_trajectory"}:
        rollout_multiplier *= 2 if algorithm == "agentic_turn" else 3
    cfg = model.config
    dim = cfg.dim
    layers = cfg.num_layers
    active_seq = seq_len + max_new_tokens
    if algorithm in {"agentic_turn", "agentic_trajectory"}:
        active_seq += max_new_tokens * (
            1 if algorithm == "agentic_turn" else 2
        )
    optimizer_bytes = optimizer_state_bytes(model, optimizer=config.optimizer, dtype_bytes=4)
    state_bytes = trainable * 8 + optimizer_bytes + ref_params * 4
    activation_bytes = batch_size * rollout_multiplier * active_seq * dim * layers * 2 * 7
    overhead_bytes = max(0.5 * 1024 ** 3, 0.10 * (state_bytes + activation_bytes))
    estimate_gb = (state_bytes + activation_bytes + overhead_bytes) / 1024 ** 3

    print("\nRL budget summary")
    print(f"  model parameters: {params:,}")
    print(f"  trainable parameters: {trainable:,}")
    print(f"  reference parameters: {ref_params:,}")
    print(f"  sequence length: {seq_len}")
    print(f"  max new tokens: {max_new_tokens}")
    print(f"  batch size: {batch_size}")
    print(f"  rollout multiplier: {rollout_multiplier}")
    print(f"  optimizer: {config.optimizer}")
    print(f"  rough estimated VRAM: {estimate_gb:.2f} GB / {budget_gb:.1f} GB")
    print("  actual peak memory will be written to run_metrics.json\n")


base_config = dict(
    max_steps=args.max_steps,
    warmup_steps=args.warmup_steps,
    batch_size=args.batch_size,
    lr=args.lr,
    muon_lr=muon_lr,
    soft_muon_power=soft_muon_power,
    optimizer=args.optimizer,
    log_every=50,
    eval_every=0,
    save_every=resolve_save_every(args.save_every, args.max_steps),
    save_dir=args.save_dir or f"checkpoints/{args.algorithm}",
    resume_from=args.resume_from,
    initial_checkpoint=model_path,
    seed=args.seed,
    rl_metrics_every=args.rl_metrics_every,
    rl_trace_samples=args.rl_trace_samples,
)
if args.algorithm == "ppo":
    tc = PPOTrainConfig(
        ppo_max_new_tokens=args.max_new_tokens,
        ppo_inner_epochs=inner_epochs,
        ppo_kl_coef=kl_coef,
        ppo_clip_ratio=clip_ratio,
        ppo_value_clip=value_clip,
        ppo_value_coef=value_coef,
        ppo_entropy_coef=entropy_coef,
        ppo_lam=gae_lambda,
        **base_config,
    )
elif args.algorithm == "dapo":
    tc = DAPOTrainConfig(
        grpo_num_generations=num_generations,
        grpo_max_new_tokens=args.max_new_tokens,
        grpo_inner_epochs=inner_epochs,
        grpo_kl_coef=kl_coef,
        dapo_clip_ratio_low=clip_ratio_low,
        dapo_clip_ratio_high=clip_ratio_high,
        dapo_safe_length=safe_length,
        dapo_length_penalty=length_penalty,
        dapo_max_resample=max_resample,
        **base_config,
    )
else:
    config_cls = {
        "grpo": GRPOTrainConfig,
        "drgrpo": DrGRPOTrainConfig,
        "grpo_lite": GRPOLiteTrainConfig,
        "agentic_turn": AgenticTurnTrainConfig,
        "agentic_trajectory": AgenticTrajectoryTrainConfig,
        "gspo": GSPOTrainConfig,
        "rloo": RLOOTrainConfig,
        "tpo": TPOTrainConfig,
        "tpo_no_anchor": TPOTrainConfig,
        "group_pg": GroupPGTrainConfig,
        "vpo": VPOTrainConfig,
        "dg": SequenceDGTrainConfig,
        "kondo": SequenceDGTrainConfig,
        "uncertainty_dg": SequenceDGTrainConfig,
        "filtered_dg": SequenceDGTrainConfig,
        "reward_variance_dg": SequenceDGTrainConfig,
        "aspo": SequenceDGTrainConfig,
        "r2vpo": SequenceDGTrainConfig,
        "replay_dg": SequenceDGTrainConfig,
        "fresh_dg": SequenceDGTrainConfig,
    }[args.algorithm]
    policy_kwargs = dict(
        grpo_num_generations=num_generations,
        grpo_max_new_tokens=args.max_new_tokens,
        grpo_inner_epochs=inner_epochs,
        grpo_kl_coef=kl_coef,
    )
    if args.algorithm in {"grpo", "drgrpo", "gspo"}:
        policy_kwargs["grpo_clip_ratio"] = clip_ratio
    if args.algorithm == "agentic_trajectory":
        policy_kwargs.update(
            agentic_advantage_estimator=agentic_advantage_estimator,
            agentic_loss_normalizer=agentic_loss_normalizer,
        )
    if args.algorithm in {"tpo", "tpo_no_anchor"}:
        policy_kwargs["tpo_eta"] = tpo_eta
        policy_kwargs["tpo_anchor_old_policy"] = args.algorithm != "tpo_no_anchor"
    if args.algorithm == "vpo":
        policy_kwargs["vpo_num_candidates"] = vpo_num_candidates
        policy_kwargs["vpo_num_scalarizations"] = vpo_num_scalarizations
        policy_kwargs["vpo_dirichlet_alpha"] = vpo_dirichlet_alpha
        policy_kwargs["grpo_clip_ratio"] = clip_ratio
    if args.algorithm in {
        "dg", "kondo", "uncertainty_dg", "filtered_dg", "reward_variance_dg",
        "aspo", "r2vpo", "replay_dg", "fresh_dg",
    }:
        policy_kwargs.update(
            dg_eta=dg_eta,
            dg_keep_ratio=dg_keep_ratio,
            dg_uncertainty_threshold=dg_uncertainty_threshold,
            dg_replay_capacity=resolve_default(
                args.replay_capacity,
                max(8, staleness_delay + 1) if args.algorithm in {"replay_dg", "fresh_dg"} else 0,
            ),
            dg_replay_min_age=replay_min_age,
            dg_replay_age_decay=replay_age_decay,
            dg_staleness_delay=staleness_delay,
            dg_drop_stale_after=drop_stale_after,
        )
    tc = config_cls(**policy_kwargs, **base_config)
dataset_desc = {
    "name": args.task,
    "split": "train",
    "algorithm": args.algorithm,
    "max_examples": args.max_examples,
}
if args.task == "agentic_calculator":
    dataset_desc["target_contract"] = "exact_arguments_v1"
    dataset_desc["suite"] = agentic_suite
    dataset_desc["start"] = agentic_start
    if agentic_suite == "mixed":
        dataset_desc["suite_contract"] = AGENTIC_CALCULATOR_MIXTURE_CONTRACT
elif args.task == "calculator_trajectory":
    dataset_desc["target_contract"] = "two_tool_exact_arguments_v1"
    dataset_desc["start"] = agentic_start
if args.task == "reasoning_gym":
    dataset_desc.update({
        "reasoning_task": reasoning_task,
        "reasoning_config": reasoning_config,
        "reasoning_train_seed": reasoning_train_seed,
    })
sig = run_signature(tok, dataset_desc, args.seq_len)
trainer_cls = {
    "ppo": PPOTrainer,
    "grpo": GRPOTrainer,
    "drgrpo": DrGRPOTrainer,
    "grpo_lite": GRPOLiteTrainer,
    "agentic_turn": AgenticTurnTrainer,
    "agentic_trajectory": AgenticTrajectoryTrainer,
    "dapo": DAPOTrainer,
    "gspo": GSPOTrainer,
    "rloo": RLOOTrainer,
    "tpo": TPOTrainer,
    "tpo_no_anchor": TPONoAnchorTrainer,
    "group_pg": GroupPGTrainer,
    "vpo": VPOTrainer,
    "dg": DGTrainer,
    "kondo": KondoTrainer,
    "uncertainty_dg": UncertaintyDGTrainer,
    "filtered_dg": FilteredDGTrainer,
    "reward_variance_dg": RewardVarianceDGTrainer,
    "aspo": ASPOTrainer,
    "r2vpo": R2VPOTrainer,
    "replay_dg": ReplayDGTrainer,
    "fresh_dg": FreshDGTrainer,
}[args.algorithm]
trainer_kwargs = dict(signature=sig, tokenizer_sig=tokenizer_signature(tok))
if online_rl_uses_reference(args.algorithm):
    trainer_kwargs["ref_model_path"] = ref_path
if args.algorithm in agentic_algorithm_tasks:
    trainer = trainer_cls(
        model,
        train_ds,
        tc,
        tokenizer=tok,
        targets=train_ds.targets,
        **trainer_kwargs,
    )
else:
    trainer = trainer_cls(model, math_reward, train_ds, tc, **trainer_kwargs)
print_rl_budget(model, args.algorithm, args.seq_len, args.batch_size, num_generations, args.max_new_tokens, tc, args.vram_budget_gb)
trainer.train()
model = trainer.model

# Evaluate on the held-out test split — the training-set loop below was an optimistic
# debugging signal, not a paper-safe number.
print(f"\n--- After {args.algorithm.upper()} (held-out {args.task} test) ---")
model.eval()
correct = 0
eval_component_totals = {}
total = len(eval_ds)
for i in range(total):
    row = eval_ds[i]
    plen = int(row["prompt_len"].item())
    prompt_ids = row["prompt_ids"][:plen].unsqueeze(0)
    if args.algorithm in agentic_algorithm_tasks:
        target = eval_ds.targets[i]
        expected = target.result
        raw_prompt_len = int(row["raw_prompt_len"].item())
        user_prompt = tok.decode(
            row["raw_prompt_ids"][:raw_prompt_len].tolist()
        )
        turn_fn = (
            calculator_agent_turn
            if args.algorithm == "agentic_turn"
            else calculator_trajectory_turn
        )
        turn = turn_fn(
            model, tok, prompt_ids, user_prompt, target,
            args.max_new_tokens, temperature=0,
        )
        hit = turn.components["reward"]
        for name, value in turn.components.items():
            if name != "reward":
                eval_component_totals[name] = (
                    eval_component_totals.get(name, 0.0) + float(value)
                )
        text = tok.decode(turn.answer_tokens.tolist())
        predicted = text[:40]
    else:
        expected = eval_ds.answers[i]
        out = generate(model, prompt_ids, max_new_tokens=args.max_new_tokens, temperature=0)
        completion = out[:, plen:]
        completion_mask = torch.ones_like(completion, dtype=torch.bool)
    if args.task == "gsm8k":
        text = tok.decode(completion[0].tolist())
        predicted = extract_answer(text)
        hit = gsm8k_reward(text, expected)
    elif args.algorithm not in agentic_algorithm_tasks:
        batch = {"idx": torch.tensor([i])}
        result = task_reward(tok, eval_ds.answers, batch, completion, completion_mask)
        hit = float(result["reward"][0].item())
        for name, value in result.items():
            if name != "reward":
                eval_component_totals[name] = (
                    eval_component_totals.get(name, 0.0) + float(value[0].item())
                )
        text = tok.decode(completion[0].tolist())
        predicted = text[:40]
    correct += hit
    if i < 5:
        print(f"  Q: {tok.decode(prompt_ids[0].tolist())[:80]}...")
        print(f"  A: {text[:80]}  (predicted={predicted}, expected={expected}, {'OK' if hit else 'WRONG'})\n")

label = args.task if args.eval_examples == 0 else f"{args.task} subset ({total})"
if args.task == "verifier_hacking_code_repair":
    label += " proxy reward"
print(f"{label} accuracy: {correct}/{total} = {correct/total:.1%}")
if eval_component_totals:
    components = ", ".join(
        f"{name}={value / total:.1%}"
        for name, value in sorted(eval_component_totals.items())
    )
    print(f"Held-out reward components: {components}")
