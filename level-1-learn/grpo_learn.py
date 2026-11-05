"""GRPO advantage computation, teaching code. Level 1, Learn, P10.

GRPO samples a group of responses for one prompt, scores each with a
reward, and uses the group mean reward as the baseline for that prompt.
The advantage of a response is its reward minus the group mean. The
normalized form divides by the group standard deviation.

This replaces the learned value function (critic) that PPO uses for its
baseline. The cost is sampling: a group of responses per prompt is
needed before any advantage exists, and the baseline is only as stable
as the group is large.

No policy is trained here. This module computes advantages only, on
stated reward lists, so every number can be checked by hand.
"""
import numpy as np


def group_advantages(rewards):
    """Return baseline, raw advantages, std, and normalized advantages.

    The standard deviation is the population standard deviation
    (ddof 0). If every reward in the group is equal, the standard
    deviation is 0, every raw advantage is 0, and the normalized
    advantages are defined as all 0 instead of a division by zero.
    """
    r = np.asarray(rewards, dtype=float)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("rewards must be a non-empty 1D sequence")
    baseline = float(r.mean())
    advantages = r - baseline
    std = float(r.std())
    if std == 0.0:
        normalized = np.zeros_like(advantages)
    else:
        normalized = advantages / std
    return {
        "rewards": [float(v) for v in r],
        "group_size": int(r.size),
        "baseline_mean": round(baseline, 4),
        "std_population": round(std, 4),
        "advantages": [round(float(v), 4) for v in advantages],
        "advantages_normalized": [round(float(v), 4) for v in normalized],
    }


# Worked examples used in results_learn.json. Rewards are stated
# teaching values, for example a checker score of 0 or 1, plus one
# graded group with a reward of 2 for a fully correct answer.
EXAMPLE_GROUPS = {
    "prompt_a_binary": [1.0, 0.0, 1.0, 0.0],
    "prompt_b_graded": [2.0, 1.0, 0.0, 1.0],
    "prompt_c_all_equal": [1.0, 1.0, 1.0, 1.0],
}


def run_grpo_examples():
    return {name: group_advantages(rewards)
            for name, rewards in EXAMPLE_GROUPS.items()}
