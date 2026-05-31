"""Local pedagogy, tree-rollout, and OPD mechanism contracts.

These helpers keep the objective-level invariants from the corresponding
`rl-experiments` lanes available inside Minilab without importing a production
agent stack.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minilab.checks import require


def gather_log_probs(log_probs, actions):
    require(log_probs.shape[:-1] == actions.shape, "log_probs prefix shape must match actions")
    require(actions.dtype == torch.long, "actions must be integer token ids")
    require(log_probs.size(-1) > 0, "log_probs vocabulary dimension must be non-empty")
    require(((actions >= 0) & (actions < log_probs.size(-1))).all(), (
        "actions must be in the token vocabulary range"
    ))
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def spike_learnability(student_logits, actions, beta, learnability_lambda):
    """Pedagogy-sandbox spike learnability gate.

    A teacher rollout is useful only if it is both task-correct and not too far
    from the student's current local support. The spike penalty uses the gap
    between the best student token and the teacher-chosen token at each step,
    with a smooth max over the sequence.
    """
    require(beta > 0, "spike_beta must be > 0")
    require(learnability_lambda >= 0, "learnability_lambda must be non-negative")
    require(actions.size(-1) > 0, "spike_learnability requires at least one action")
    log_probs = F.log_softmax(student_logits.float(), dim=-1)
    chosen_logp = gather_log_probs(log_probs, actions)
    best_logp = log_probs.max(dim=-1).values
    surprise_gap = best_logp - chosen_logp
    spike_penalty = (
        torch.logsumexp(beta * surprise_gap, dim=-1)
        - math.log(surprise_gap.size(-1))
    ) / beta
    learnability = torch.exp(-learnability_lambda * spike_penalty)
    diagnostics = {
        "student_logp": chosen_logp,
        "avg_surprisal": -chosen_logp.mean(dim=-1),
        "avg_surprise_gap": surprise_gap.mean(dim=-1),
        "max_surprise_gap": surprise_gap.max(dim=-1).values,
        "spike_penalty": spike_penalty,
        "learnability": learnability,
    }
    return learnability, diagnostics


def pedagogical_rewards(task_rewards, student_logits, actions, beta, learnability_lambda):
    learnability, diagnostics = spike_learnability(
        student_logits, actions, beta, learnability_lambda)
    require(task_rewards.shape == learnability.shape, (
        "task_rewards must match the rollout batch shape"
    ))
    rewards = task_rewards.float() * learnability
    return rewards, {
        "task_reward": task_rewards.float().mean().item(),
        "pedagogy_reward": rewards.mean().item(),
        "learnability": diagnostics["learnability"].mean().item(),
        "spike_penalty": diagnostics["spike_penalty"].mean().item(),
        "avg_surprise_gap": diagnostics["avg_surprise_gap"].mean().item(),
        "max_surprise_gap": diagnostics["max_surprise_gap"].mean().item(),
        "avg_student_surprisal": diagnostics["avg_surprisal"].mean().item(),
    }


def gated_imitation_weights(student_logits, actions, gate_kappa, gate_gamma):
    """Return the gated-imitation weights used by the pedagogy sandbox."""
    require(gate_kappa > 0, "gate_kappa must be > 0")
    require(actions.size(-1) > 0, "gated imitation requires at least one action")
    log_probs = F.log_softmax(student_logits, dim=-1)
    logp_a = gather_log_probs(log_probs, actions)
    return torch.sigmoid(gate_kappa * (logp_a.detach() - gate_gamma)), logp_a


def gated_imitation_loss(student_logits, actions, gate_kappa, gate_gamma):
    weights, logp_a = gated_imitation_weights(
        student_logits, actions, gate_kappa, gate_gamma)
    denom = weights.sum(dim=-1).clamp(min=1e-8)
    loss = ((weights * -logp_a).sum(dim=-1) / denom).mean()
    return loss, {
        "assim_loss": loss.item(),
        "assim_gate_mean": weights.mean().item(),
        "assim_logp_mean": logp_a.mean().item(),
    }


@dataclass(frozen=True)
class TreeSegment:
    segment_id: int
    parent_id: int
    token_count: int


def tree_credit_weights(segments, root_advantage):
    """Child-count-normalized root credit for recursive rollout trees.

    Segments split their inherited credit by sibling count. This prevents a
    highly branching rollout from dominating a batch only because it generated
    more child segments.
    """
    require(segments, "tree credit requires at least one segment")
    by_id = {segment.segment_id: segment for segment in segments}
    require(len(by_id) == len(segments), "tree segment ids must be unique")
    by_parent = {}
    for segment in segments:
        require(segment.token_count > 0, "tree segment token_count must be > 0")
        by_parent.setdefault(segment.parent_id, []).append(segment.segment_id)

    inherited = {}
    weights = {}
    visiting = set()

    def visit(segment):
        if segment.segment_id in inherited:
            return inherited[segment.segment_id]
        require(segment.segment_id not in visiting, "tree segments must not contain cycles")
        visiting.add(segment.segment_id)
        if segment.parent_id < 0:
            parent_credit = root_advantage
        else:
            require(segment.parent_id in by_id, "tree segment parent is missing")
            parent_credit = visit(by_id[segment.parent_id])
        siblings = by_parent.get(segment.parent_id, [segment.segment_id])
        credit = parent_credit / len(siblings)
        inherited[segment.segment_id] = credit
        weights[segment.segment_id] = credit / segment.token_count
        visiting.remove(segment.segment_id)
        return credit

    for segment in segments:
        visit(segment)
    return weights


def rlm_segment_weights(root_segment_count, child_rollout_segment_counts, train_child_trajectories):
    """RLM tree weighting contract from `rlm_grpo`.

    Root generated turns share one root budget. If child trajectories are
    trained, each child rollout receives one child share and splits it across
    that child's generated turns.
    """
    require(root_segment_count > 0, "RLM rollouts must contain at least one root segment")
    root_weight = 1.0 / root_segment_count
    if not train_child_trajectories or not child_rollout_segment_counts:
        return root_weight, []
    child_count = len(child_rollout_segment_counts)
    child_weights = []
    for segment_count in child_rollout_segment_counts:
        require(segment_count > 0, "child rollouts must contain at least one segment")
        child_weights.extend([1.0 / child_count / segment_count] * segment_count)
    return root_weight, child_weights


def tree_policy_loss(segment_logps, segments, root_advantage):
    weights = tree_credit_weights(segments, root_advantage)
    losses = []
    for segment in segments:
        require(segment.segment_id in segment_logps, "missing segment log-prob")
        losses.append(-segment_logps[segment.segment_id] * weights[segment.segment_id])
    return torch.stack(losses).mean()


def topk_support_mask(student_logits, teacher_logits, k, support):
    require(k > 0, "top-k support requires k > 0")
    require(k <= student_logits.size(-1), "top-k support cannot exceed the vocabulary size")
    require(student_logits.shape == teacher_logits.shape, "student and teacher logits must have the same shape")
    require(support in {"student", "teacher", "intersection"}, "unknown OPD support")
    student_top = student_logits.topk(k, dim=-1).indices
    teacher_top = teacher_logits.topk(k, dim=-1).indices
    mask = torch.zeros_like(student_logits, dtype=torch.bool)
    if support == "student":
        return mask.scatter(-1, student_top, True)
    if support == "teacher":
        return mask.scatter(-1, teacher_top, True)
    student_mask = mask.scatter(-1, student_top, True)
    teacher_mask = torch.zeros_like(student_logits, dtype=torch.bool).scatter(-1, teacher_top, True)
    return student_mask & teacher_mask


def opd_reverse_kl(student_logits, teacher_logits, mask=None):
    """Full-vocab or unnormalized top-k reverse KL.

    The top-k OPD sandbox deliberately does not renormalize over the retained
    support; omitted teacher mass should remain omitted so support failure is
    visible in the objective.
    """
    require(student_logits.shape == teacher_logits.shape, (
        "student and teacher logits must have the same shape"
    ))
    require(student_logits.size(-1) > 0, "OPD logits vocabulary dimension must be non-empty")
    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    student_p = student_logp.exp()
    if mask is not None:
        require(mask.shape == student_logits.shape, "OPD support mask must match logits shape")
        require(mask.dtype == torch.bool, "OPD support mask must be bool")
        student_p = student_p * mask
    return (student_p * (student_logp - teacher_logp)).sum(dim=-1).mean()
