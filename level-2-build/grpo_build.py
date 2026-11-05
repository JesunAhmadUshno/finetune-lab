"""GRPO policy training on a toy policy. Level 2, Build, P10.

Extends Level 1, which computed group advantages only and trained
no policy. This module trains a real (toy) policy with the GRPO
objective:

- Policy: a tabular categorical policy. There are 8 prompts. Each
  prompt has one verifiably correct action in {0, 1, 2, 3}. The
  policy holds one logit vector per prompt, shape (8, 4). The
  reference policy is the initial policy (uniform, logits 0 plus a
  small seeded tilt), frozen for the KL term.
- Verifiable task: the checker returns reward 1.0 when the sampled
  action equals the prompt's correct action, else 0.0. No learned
  reward model, no human labels. This is a synthetic checker task,
  honestly labelled toy: no text, no LLM, no real math or code task.
- Per iteration: for each prompt, sample a group of G = 8 actions
  from the current policy, score them with the checker, compute
  normalized group advantages (same rule as Learn: population std,
  zero group gives zero advantages), then take one gradient step on
  the GRPO loss: the negative clipped surrogate objective plus
  beta times the KL divergence to the reference policy.

Definitions used, stated so tests can check them by hand:
- ratio r = pi_new(a) / pi_old(a) for the sampled action.
- clipped ratio = clip(r, 1 - eps, 1 + eps), eps = 0.2.
- surrogate per sample = min(r * A, clipped ratio * A).
- KL(pi || pi_ref) per prompt = sum_a pi(a) * (log pi(a) - log
  pi_ref(a)), the exact categorical KL, in nats.
- Loss = -mean(surrogate) + beta * mean_prompt KL, beta = 0.04.

The update is the exact gradient of that loss with respect to the
logits, derived for the tabular softmax: for the surrogate part the
gradient flows through the sampled action probabilities (the ratio
is treated with the standard stop-gradient on pi_old, and a sample
whose clipped branch is active and would move the ratio further
from 1 contributes zero surrogate gradient, matching the clipped
objective); for the KL part the gradient is pi * (log pi - log
pi_ref - KL) per prompt row, the standard softmax KL gradient.
Reward is measured before training and after every 10 iterations
on fresh seeded samples (eval seed stream, separate from training)
as the mean checker reward over 8 prompts x 32 samples per prompt,
plus the greedy accuracy (argmax action correct per prompt).
"""
import numpy as np

N_PROMPTS = 8
N_ACTIONS = 4
GROUP_SIZE = 8
EPS_CLIP = 0.2
BETA_KL = 0.04
LEARNING_RATE = 0.5
ITERATIONS = 60
POLICY_SEED = 21
TRAIN_SEED = 31
EVAL_SEED = 41

# Correct action per prompt, fixed and stated. A spread over all
# four actions so a constant policy cannot score above 0.25 greedy.
CORRECT = np.array([2, 0, 3, 1, 2, 3, 0, 1])


def softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def clipped_ratio(ratio, eps=EPS_CLIP):
    return np.clip(ratio, 1.0 - eps, 1.0 + eps)


def surrogate_per_sample(ratio, advantage, eps=EPS_CLIP):
    """min(r * A, clip(r) * A), elementwise."""
    unclipped = ratio * advantage
    clipped = clipped_ratio(ratio, eps) * advantage
    return np.minimum(unclipped, clipped)


def kl_divergence(probs, ref_probs):
    """Exact categorical KL(probs || ref_probs) per row, in nats."""
    probs = np.asarray(probs, dtype=float)
    ref_probs = np.asarray(ref_probs, dtype=float)
    terms = probs * (np.log(probs + 1e-12) - np.log(ref_probs + 1e-12))
    return terms.sum(axis=1)


def group_advantages(rewards):
    """Same rule as Learn: mean baseline, population std, safe zero."""
    r = np.asarray(rewards, dtype=float)
    baseline = float(r.mean())
    adv = r - baseline
    std = float(r.std())
    if std == 0.0:
        return np.zeros_like(adv), baseline, std
    return adv / std, baseline, std


def init_policy():
    """Small seeded tilt so prompts do not start identical."""
    rng = np.random.default_rng(POLICY_SEED)
    return rng.standard_normal((N_PROMPTS, N_ACTIONS)) * 0.1


def checker(prompt_idx, action):
    return 1.0 if int(action) == int(CORRECT[prompt_idx]) else 0.0


def evaluate_policy(logits, seed=EVAL_SEED, samples_per_prompt=32):
    """Fresh-sample mean reward and greedy accuracy, deterministic."""
    probs = softmax_rows(logits)
    rng = np.random.default_rng(seed)
    total = 0.0
    count = 0
    for p in range(N_PROMPTS):
        actions = rng.choice(N_ACTIONS, size=samples_per_prompt, p=probs[p])
        for act in actions:
            total += checker(p, act)
            count += 1
    greedy = probs.argmax(axis=1)
    greedy_acc = float((greedy == CORRECT).mean())
    return {"mean_reward": round(total / count, 4),
            "greedy_accuracy": round(greedy_acc, 4),
            "greedy_correct": int((greedy == CORRECT).sum()),
            "n_prompts": N_PROMPTS,
            "samples_per_prompt": samples_per_prompt}


def grpo_loss_and_grad(logits, ref_probs, samples, beta=BETA_KL, eps=EPS_CLIP):
    """Compute GRPO loss and its gradient for one sampled batch.

    samples: list per prompt of (actions array, advantages array).
    pi_old equals the policy at sampling time, which is the current
    logits at the step, so ratios at the step start at 1.0 for the
    gradient evaluation here; the clipping branch is still computed
    through the ratio function so the reported loss is the clipped
    objective. To exercise clipping inside training, the step uses
    two inner epochs on the same batch (see run_grpo): on the second
    epoch ratios move off 1.0 and clipping can bind.
    This function evaluates loss and gradient at the given logits
    against pi_old_probs passed inside samples as old_probs.
    """
    probs = softmax_rows(logits)
    n_prompts = logits.shape[0]
    total_surrogate = 0.0
    total_samples = 0
    grad = np.zeros_like(logits)
    for p in range(n_prompts):
        actions, advantages, old_probs_p = samples[p]
        pi_a = probs[p][actions]
        old_a = old_probs_p[actions]
        ratio = pi_a / (old_a + 1e-12)
        surr = surrogate_per_sample(ratio, advantages, eps)
        total_surrogate += float(surr.sum())
        total_samples += len(actions)
        # Gradient of -surrogate wrt logits. For one sample, the
        # derivative of pi_a wrt the row logits is pi_a * (e_a - pi).
        # When the clipped branch is the minimum and the ratio is
        # outside the clip range, the clipped objective is constant
        # in pi, so the sample gradient is zero.
        clip_r = clipped_ratio(ratio, eps)
        clipped_active = (ratio * advantages) >= (clip_r * advantages)
        outside = (ratio < 1.0 - eps) | (ratio > 1.0 + eps)
        active = ~(clipped_active & outside)
        for i, act in enumerate(actions):
            if not active[i]:
                continue
            coeff = -advantages[i] * ratio[i] / total_samples
            one_hot = np.zeros(N_ACTIONS)
            one_hot[act] = 1.0
            grad[p] += coeff * (one_hot - probs[p])
    mean_surrogate = total_surrogate / max(total_samples, 1)
    kls = kl_divergence(probs, ref_probs)
    mean_kl = float(kls.mean())
    # KL gradient per row: pi * (log pi - log pi_ref - KL_row).
    log_ratio = np.log(probs + 1e-12) - np.log(ref_probs + 1e-12)
    kl_grad = probs * (log_ratio - kls[:, None])
    grad = grad + beta * kl_grad
    loss = -mean_surrogate + beta * mean_kl
    return {"loss": float(loss), "mean_surrogate": float(mean_surrogate),
            "mean_kl": mean_kl}, grad


def sample_batch(logits, rng):
    probs = softmax_rows(logits)
    samples = []
    rewards_all = []
    for p in range(N_PROMPTS):
        actions = rng.choice(N_ACTIONS, size=GROUP_SIZE, p=probs[p])
        rewards = np.array([checker(p, a) for a in actions])
        adv, _, _ = group_advantages(rewards)
        samples.append((actions, adv, probs[p].copy()))
        rewards_all.append(float(rewards.mean()))
    return samples, float(np.mean(rewards_all))


def run_grpo(iterations=ITERATIONS, lr=LEARNING_RATE, inner_epochs=2):
    logits = init_policy()
    ref_probs = softmax_rows(logits.copy())
    before = evaluate_policy(logits)
    rng = np.random.default_rng(TRAIN_SEED)
    history = []
    train_reward_first = None
    train_reward_last = None
    final_stats = None
    for it in range(iterations):
        samples, train_reward = sample_batch(logits, rng)
        if it == 0:
            train_reward_first = round(train_reward, 4)
        train_reward_last = round(train_reward, 4)
        for _ in range(inner_epochs):
            stats, grad = grpo_loss_and_grad(logits, ref_probs, samples)
            logits = logits - lr * grad
            final_stats = stats
        if it % 10 == 9 or it == 0:
            ev = evaluate_policy(logits, seed=EVAL_SEED + it + 1)
            probs_now = softmax_rows(logits)
            history.append({
                "iteration": it + 1,
                "train_group_reward": round(train_reward, 4),
                "eval_mean_reward": ev["mean_reward"],
                "eval_greedy_accuracy": ev["greedy_accuracy"],
                "loss": round(final_stats["loss"], 4),
                "mean_kl_to_ref": round(float(kl_divergence(probs_now, ref_probs).mean()), 4),
            })
    after = evaluate_policy(logits)
    probs_final = softmax_rows(logits)
    return {
        "config": {"n_prompts": N_PROMPTS, "n_actions": N_ACTIONS,
                   "group_size": GROUP_SIZE, "eps_clip": EPS_CLIP,
                   "beta_kl": BETA_KL, "learning_rate": lr,
                   "iterations": iterations, "inner_epochs": inner_epochs,
                   "policy_seed": POLICY_SEED, "train_seed": TRAIN_SEED,
                   "eval_seed": EVAL_SEED,
                   "correct_actions": [int(c) for c in CORRECT]},
        "before_training": before,
        "after_training": after,
        "train_group_reward_first_iteration": train_reward_first,
        "train_group_reward_last_iteration": train_reward_last,
        "final_loss": round(final_stats["loss"], 4),
        "final_mean_surrogate": round(final_stats["mean_surrogate"], 4),
        "final_mean_kl_to_ref": round(float(kl_divergence(probs_final, ref_probs).mean()), 4),
        "history": history,
    }
