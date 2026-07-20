"""Local online-RL diagnostics for laptop-scale post-training runs."""

import json
from pathlib import Path

import torch

from minilab.checks import require


SCHEMA = "minilab.online_rl.v1"


def append_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def split_reward_result(result):
    if torch.is_tensor(result):
        require(result.dim() == 1, "reward tensor must have shape [batch]")
        return result, {}
    require(isinstance(result, dict), "reward_fn must return a tensor or a dict with a 'reward' tensor")
    require("reward" in result, "reward component dict must contain a 'reward' tensor")
    reward = result["reward"]
    require(torch.is_tensor(reward), "reward component 'reward' must be a tensor")
    require(reward.dim() == 1, "reward component 'reward' must have shape [batch]")
    components = {
        key: value
        for key, value in result.items()
        if key != "reward"
    }
    for key, value in components.items():
        require(torch.is_tensor(value), f"reward component '{key}' must be a tensor")
        require(value.shape == reward.shape, (
            f"reward component '{key}' must match reward shape"
        ))
    return reward, components


def stack_reward_results(results, device):
    require(len(results) > 0, "reward stacking requires at least one reward result")
    rewards = []
    component_names = None
    reward_shape = None
    stacked_components = {}
    for result in results:
        reward, components = split_reward_result(result)
        reward_shape = reward.shape if reward_shape is None else reward_shape
        require(reward.shape == reward_shape, "reward tensors must have the same batch shape")
        rewards.append(reward.to(device))
        names = set(components)
        component_names = names if component_names is None else component_names
        require(names == component_names, "reward component keys must match across rollout generations")
        for key, value in components.items():
            stacked_components.setdefault(key, []).append(value.to(device))

    reward_tensor = torch.stack(rewards, dim=1)
    component_tensors = {
        key: torch.stack(values, dim=1)
        for key, values in stacked_components.items()
    }
    return reward_tensor, component_tensors


def group_stats(rewards, adv, completion_masks=None, components=None):
    require(rewards.dim() == 2, "group diagnostics expect rewards shaped [batch, generations]")
    require(rewards.size(0) > 0 and rewards.size(1) > 0, "group diagnostics require non-empty rewards")
    require(adv.shape == rewards.shape, "group diagnostics advantages must match rewards shape")
    std = rewards.std(dim=1, unbiased=False)
    metrics = {
        "reward_mean": float(rewards.mean().item()),
        "reward_std_mean": float(std.mean().item()),
        "mixed_group_rate": float((std > 0).float().mean().item()),
        "all_zero_group_rate": float(rewards.eq(0).all(dim=1).float().mean().item()),
        "all_equal_group_rate": float(std.eq(0).float().mean().item()),
        "adv_abs_mean": float(adv.abs().mean().item()),
    }
    if completion_masks is not None:
        require(isinstance(completion_masks, (list, tuple)), "completion masks must be a list or tuple")
        require(len(completion_masks) == rewards.size(1), (
            "completion mask count must match reward generations"
        ))
        lengths = torch.stack([
            _completion_lengths(mask, rewards.size(0))
            for mask in completion_masks
        ], dim=1)
        metrics["completion_length_mean"] = float(lengths.mean().item())
        metrics["completion_length_max"] = float(lengths.max().item())
    for key, value in (components or {}).items():
        require(value.shape == rewards.shape, f"reward component '{key}' must match rewards shape")
        metrics[f"reward_component_{key}_mean"] = float(value.float().mean().item())
    return metrics


def ppo_stats(reward, advantages, mask):
    require(reward.dim() == 1, "PPO diagnostics expect reward shaped [batch]")
    require(reward.numel() > 0, "PPO diagnostics require at least one reward")
    require(mask.dim() == 2, "PPO diagnostics mask must have shape [batch, tokens]")
    require(reward.size(0) == mask.size(0), "PPO diagnostics reward batch must match mask batch")
    require(advantages.shape == mask.shape, "PPO diagnostics advantages must match mask shape")
    require(mask.dtype == torch.bool or mask.dtype.is_floating_point, "PPO diagnostics mask must be bool or float")
    lengths = mask.sum(dim=1).float()
    require((lengths > 0).all(), "PPO diagnostics require at least one completion token per row")
    return {
        "reward_mean": float(reward.mean().item()),
        "reward_std_mean": float(reward.std(unbiased=False).item()) if reward.numel() > 1 else 0.0,
        "adv_abs_mean": float(advantages.abs().mean().item()),
        "completion_length_mean": float(lengths.mean().item()),
        "completion_length_max": float(lengths.max().item()),
    }


def _completion_lengths(mask, batch_size):
    require(torch.is_tensor(mask), "completion masks must contain tensors")
    require(mask.dim() == 2, "completion masks must have shape [batch, tokens]")
    require(mask.size(0) == batch_size, "completion mask batch size must match rewards")
    require(mask.dtype == torch.bool, "completion masks must be bool tensors")
    lengths = mask.sum(dim=1).to(torch.float32)
    require((lengths > 0).all(), "completion masks require at least one token per row")
    return lengths


def rollout_system_stats(rollout):
    metrics = {
        "generation_latency_ms": float(rollout.generation_latency_ms),
        "reward_latency_ms": float(rollout.reward_latency_ms),
        "generated_tokens": int(rollout.generated_tokens),
        "reward_calls": int(rollout.reward_calls),
        "queue_depth": int(rollout.queue_depth),
        "dropped_stale_count": int(rollout.dropped_stale_count),
        "staleness_delay": int(rollout.staleness_delay),
        "drop_stale_after": int(rollout.drop_stale_after),
        "trajectory_age": int(rollout.age),
    }
    if metrics.get("generation_latency_ms", 0) > 0 and metrics.get("generated_tokens", 0) > 0:
        metrics["generation_tokens_per_second"] = (
            1000.0 * metrics["generated_tokens"] / metrics["generation_latency_ms"]
        )
    return metrics


def scalar_record(trainer_name, step, algorithm, metrics):
    record = {
        "schema": SCHEMA,
        "kind": "metrics",
        "trainer": trainer_name,
        "algorithm": algorithm,
        "step": int(step),
    }
    record.update(metrics)
    return record


def rollout_records(trainer_name, step, algorithm, batch, rollout, max_records):
    if max_records <= 0 or not rollout.completions:
        return []
    records = []
    rewards = rollout.rewards.detach().cpu()
    adv = rollout.adv.detach().cpu()
    for k, completions in enumerate(rollout.completions):
        masks = rollout.completion_masks[k]
        for b in range(completions.size(0)):
            if len(records) >= max_records:
                return records
            length = int(masks[b].sum().item())
            record = {
                "schema": SCHEMA,
                "kind": "trajectory",
                "trainer": trainer_name,
                "algorithm": algorithm,
                "step": int(step),
                "prompt_index": int(b),
                "generation_index": int(k),
                "reward": float(rewards[b, k].item()),
                "advantage": float(adv[b, k].item()),
                "completion_length": length,
                "completion_tokens": completions[b, :length].detach().cpu().tolist(),
                "accepted_for_training": bool(rollout.accepted_for_training),
                "drop_reason": rollout.drop_reason,
                "policy_version": int(step),
                "rollout_version": int(max(0, step - rollout.age)),
                "trajectory_age": int(rollout.age),
            }
            for name, values in (rollout.reward_components or {}).items():
                record[f"reward_component_{name}"] = float(values[b, k].item())
            records.append(record)
    return records


def agentic_rollout_records(trainer_name, step, algorithm, rollout, max_records):
    if max_records <= 0 or not rollout.tool_completions:
        return []
    require(len(rollout.tool_completions) == rollout.rewards.size(1), (
        "agentic tool completion count must match reward generations"
    ))
    require(len(rollout.answer_completions) == rollout.rewards.size(1), (
        "agentic answer completion count must match reward generations"
    ))
    require(len(rollout.tool_observations) == rollout.rewards.size(1), (
        "agentic observation count must match reward generations"
    ))
    records = []
    rewards = rollout.rewards.detach().cpu()
    answer_adv = rollout.adv.detach().cpu()
    tool_adv = rollout.tool_adv.detach().cpu()
    for k, tool_completions in enumerate(rollout.tool_completions):
        tool_masks = rollout.tool_completion_masks[k]
        answer_completions = rollout.answer_completions[k]
        answer_masks = rollout.answer_completion_masks[k]
        require(tool_completions.size(0) == answer_completions.size(0), (
            "agentic tool and answer completion batches must match"
        ))
        require(len(rollout.tool_observations[k]) == tool_completions.size(0), (
            "agentic observations must match their completion batch"
        ))
        for b in range(tool_completions.size(0)):
            if len(records) >= max_records:
                return records
            tool_length = int(tool_masks[b].sum().item())
            answer_length = int(answer_masks[b].sum().item())
            record = {
                "schema": SCHEMA,
                "kind": "agentic_trajectory",
                "trainer": trainer_name,
                "algorithm": algorithm,
                "step": int(step),
                "prompt_index": int(b),
                "generation_index": int(k),
                "reward": float(rewards[b, k].item()),
                "tool_advantage": float(tool_adv[b, k].item()),
                "advantage": float(answer_adv[b, k].item()),
                "tool_completion_length": tool_length,
                "answer_completion_length": answer_length,
                "completion_length": tool_length + answer_length,
                "tool_completion_tokens": (
                    tool_completions[b, :tool_length].detach().cpu().tolist()
                ),
                "answer_completion_tokens": (
                    answer_completions[b, :answer_length].detach().cpu().tolist()
                ),
                "completion_tokens": (
                    answer_completions[b, :answer_length].detach().cpu().tolist()
                ),
                "tool_observation": rollout.tool_observations[k][b],
                "accepted_for_training": True,
                "drop_reason": "",
                "policy_version": int(step),
                "rollout_version": int(step),
                "trajectory_age": 0,
            }
            for name, values in (rollout.reward_components or {}).items():
                record[f"reward_component_{name}"] = float(values[b, k].item())
            records.append(record)
    return records


def agentic_trajectory_rollout_records(
    trainer_name,
    step,
    algorithm,
    rollout,
    max_records,
    *,
    advantage_estimator="group_standardized",
    loss_normalizer="response_length",
):
    if max_records <= 0 or not rollout.first_tool_completions:
        return []
    generations = rollout.rewards.size(1)
    for name in (
        "first_tool_completions",
        "second_tool_completions",
        "answer_completions",
        "trajectory_observations",
    ):
        require(len(getattr(rollout, name)) == generations, (
            f"agentic trajectory {name} count must match reward generations"
        ))
    records = []
    rewards = rollout.rewards.detach().cpu()
    answer_adv = rollout.adv.detach().cpu()
    first_adv = rollout.first_tool_adv.detach().cpu()
    second_adv = rollout.second_tool_adv.detach().cpu()
    for k, first_completions in enumerate(rollout.first_tool_completions):
        first_masks = rollout.first_tool_completion_masks[k]
        second_completions = rollout.second_tool_completions[k]
        second_masks = rollout.second_tool_completion_masks[k]
        answer_completions = rollout.answer_completions[k]
        answer_masks = rollout.answer_completion_masks[k]
        batch_size = first_completions.size(0)
        require(
            second_completions.size(0) == batch_size
            and answer_completions.size(0) == batch_size,
            "agentic trajectory completion batches must match",
        )
        require(len(rollout.trajectory_observations[k]) == batch_size, (
            "agentic trajectory observations must match their completion batch"
        ))
        for b in range(batch_size):
            if len(records) >= max_records:
                return records
            first_length = int(first_masks[b].sum().item())
            second_length = int(second_masks[b].sum().item())
            answer_length = int(answer_masks[b].sum().item())
            first_observation, second_observation = (
                rollout.trajectory_observations[k][b]
            )
            record = {
                "schema": SCHEMA,
                "kind": "agentic_two_tool_trajectory",
                "trainer": trainer_name,
                "algorithm": algorithm,
                "advantage_estimator": advantage_estimator,
                "loss_normalizer": loss_normalizer,
                "step": int(step),
                "prompt_index": int(b),
                "generation_index": int(k),
                "reward": float(rewards[b, k].item()),
                "first_tool_advantage": float(first_adv[b, k].item()),
                "second_tool_advantage": float(second_adv[b, k].item()),
                "advantage": float(answer_adv[b, k].item()),
                "first_tool_completion_length": first_length,
                "second_tool_completion_length": second_length,
                "answer_completion_length": answer_length,
                "completion_length": (
                    first_length + second_length + answer_length
                ),
                "first_tool_completion_tokens": (
                    first_completions[b, :first_length].detach().cpu().tolist()
                ),
                "second_tool_completion_tokens": (
                    second_completions[b, :second_length].detach().cpu().tolist()
                ),
                "answer_completion_tokens": (
                    answer_completions[b, :answer_length].detach().cpu().tolist()
                ),
                "completion_tokens": (
                    answer_completions[b, :answer_length].detach().cpu().tolist()
                ),
                "first_tool_observation": first_observation,
                "second_tool_observation": second_observation,
                "accepted_for_training": True,
                "drop_reason": "",
                "policy_version": int(step),
                "rollout_version": int(step),
                "trajectory_age": 0,
            }
            for name, values in (rollout.reward_components or {}).items():
                record[f"reward_component_{name}"] = float(values[b, k].item())
            records.append(record)
    return records


def vpo_rollout_records(trainer_name, step, algorithm, rollout, max_records):
    if max_records <= 0 or not rollout.completions:
        return []
    records = []
    rewards = rollout.rewards.detach().cpu()
    adv = rollout.adv.detach().cpu()
    candidate_rewards = rollout.candidate_rewards.detach().cpu()
    K = rewards.size(1)
    C = candidate_rewards.size(2)
    flat_index = 0
    for k in range(K):
        for c in range(C):
            completions = rollout.completions[flat_index]
            masks = rollout.completion_masks[flat_index]
            for b in range(completions.size(0)):
                if len(records) >= max_records:
                    return records
                length = int(masks[b].sum().item())
                record = {
                    "schema": SCHEMA,
                    "kind": "trajectory",
                    "trainer": trainer_name,
                    "algorithm": algorithm,
                    "step": int(step),
                    "prompt_index": int(b),
                    "generation_index": int(k),
                    "candidate_index": int(c),
                    "reward": float(rewards[b, k].item()),
                    "candidate_reward": float(candidate_rewards[b, k, c].item()),
                    "advantage": float(adv[b, k].item()),
                    "completion_length": length,
                    "completion_tokens": completions[b, :length].detach().cpu().tolist(),
                    "accepted_for_training": bool(rollout.accepted_for_training),
                    "drop_reason": rollout.drop_reason,
                    "policy_version": int(step),
                    "rollout_version": int(max(0, step - rollout.age)),
                    "trajectory_age": int(rollout.age),
                }
                for name, values in rollout.reward_components.items():
                    record[f"reward_component_{name}"] = float(values[b, k, c].item())
                records.append(record)
            flat_index += 1
    return records


def gallery_sections(records):
    sections = {
        "successful": [],
        "all_zero_or_negative": [],
        "long_high_reward": [],
        "wrong_answer_good_format": [],
        "answer_without_required_format": [],
        "verifier_or_timeout_failure": [],
        "proxy_reward_hack": [],
        "metamorphic_disagreement": [],
        "stale": [],
        "malformed_or_dropped": [],
    }
    if not records:
        return sections
    positive = [r for r in records if r.get("reward", 0.0) > 0]
    length_cut = max((r.get("completion_length", 0) for r in positive), default=0)
    for record in records:
        reward = record.get("reward", 0.0)
        if not record.get("accepted_for_training", True):
            sections["malformed_or_dropped"].append(record)
        if record.get("trajectory_age", 0) > 0:
            sections["stale"].append(record)
        if record.get("reward_component_format") == 1.0 and record.get("reward_component_answer") == 0.0:
            sections["wrong_answer_good_format"].append(record)
        if record.get("reward_component_answer") == 1.0 and record.get("reward_component_format") == 0.0:
            sections["answer_without_required_format"].append(record)
        if record.get("reward_component_timeout_free") == 0.0 or record.get("reward_component_syntax") == 0.0:
            sections["verifier_or_timeout_failure"].append(record)
        if record.get("reward_component_reward_hack") == 1.0:
            sections["proxy_reward_hack"].append(record)
        if record.get("reward_component_proxy_metamorphic_disagreement") == 1.0:
            sections["metamorphic_disagreement"].append(record)
        if reward > 0:
            sections["successful"].append(record)
            if record.get("completion_length", 0) >= length_cut:
                sections["long_high_reward"].append(record)
        else:
            sections["all_zero_or_negative"].append(record)
    return sections


def write_failure_gallery(records, output_dir, limit=20):
    require(limit >= 0, "failure gallery limit must be >= 0")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = gallery_sections(records)
    lines = ["# Failure Gallery", ""]
    for name, rows in sections.items():
        lines.append(f"## {name.replace('_', ' ').title()}")
        lines.append("")
        for record in rows[:limit]:
            lines.append(
                f"- step={record.get('step')} gen={record.get('generation_index')} "
                f"reward={record.get('reward')} age={record.get('trajectory_age', 0)} "
                f"length={record.get('completion_length')}"
            )
            if "candidate_index" in record:
                lines[-1] += f" candidate={record.get('candidate_index')}"
            if record.get("drop_reason"):
                lines[-1] += f" drop_reason={record.get('drop_reason')}"
            component_parts = [
                f"{key.removeprefix('reward_component_')}={value}"
                for key, value in sorted(record.items())
                if key.startswith("reward_component_")
            ]
            if component_parts:
                lines.append(f"  components: {', '.join(component_parts)}")
            if record.get("completion_text"):
                lines.append(f"  text: {record['completion_text'][:240]}")
            tokens = record.get("completion_tokens", [])
            lines.append(f"  tokens={tokens[:80]}")
        if not rows:
            lines.append("- none")
        lines.append("")
    (output_dir / "gallery.md").write_text("\n".join(lines) + "\n")
    return sections
