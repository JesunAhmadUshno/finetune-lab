"""DPO run, forgetting checks, and a reward hacking probe.
Level 3, Deep, P10 finetune-lab.

Build code reuse: this module imports the Build GRPO module
(level-2-build/grpo_build.py) and reuses its tabular policy
definition exactly: the same init_policy (policy seed 21), the
same 8 prompts, the same correct actions, the same checker, and
the same evaluate_policy function (eval seed 41, 32 samples per
prompt). No Build file is modified. Attribution: the policy,
checker, and evaluator are from Level 2 Build, written for this
repo; the DPO code below is new in this level.

Policy: one logit vector per prompt, shape (8, 4), a categorical
softmax policy. This is a toy NumPy policy, not a language
model: actions stand in for responses, and there is no text,
no tokenizer, and no sequence. Log-probabilities are over the
4 actions of one prompt.

Preference pairs, deterministic rule, stated in full:
- The chosen response for a prompt is its checker-correct
  action (the Build CORRECT vector).
- The rejected responses are the 3 wrong actions, ordered by
  their offset from the correct action: plus 1, plus 2, plus 3
  modulo 4.
- Train pairs: 8 pairs, one per prompt, using the plus 1
  rejected action.
- Held-out pairs: 16 pairs, two per prompt, using the plus 2
  and plus 3 rejected actions. These rejected actions never
  appear as rejected in training, so held-out preference
  accuracy tests whether raising the chosen action and lowering
  one wrong action generalises to contrasts never trained.

DPO training:
- The reference policy is a frozen copy of the initial policy.
- Loss per pair: negative log sigmoid of beta times the
  reference-adjusted margin, where the margin for one pair is
  (log pi(chosen) minus log pi_ref(chosen)) minus
  (log pi(rejected) minus log pi_ref(rejected)). Beta is 0.1.
- Gradient, hand-derived: for one pair, the derivative of the
  reference-adjusted margin with respect to that prompt's
  logits is exactly (one-hot(chosen) minus one-hot(rejected)),
  because the softmax Jacobian terms cancel between the two
  log-probabilities of the same row. So the gradient of the
  mean loss is, per pair, minus beta times (1 minus sigmoid(
  beta times margin)) times (one-hot(chosen) minus
  one-hot(rejected)), averaged over pairs. The unit tests check
  this gradient against finite differences and against a
  hand-computed case.
- Optimizer: full-batch gradient descent, learning rate 0.5,
  300 steps, fixed seeds throughout. Deterministic.

Forgetting checks:
- The pre-DPO policy task is the Build GRPO checker task.
  Greedy accuracy and fresh-sample mean reward are measured
  with the Build evaluator before and after DPO, and the exact
  change is reported, whatever its sign.
- Drift: mean exact categorical KL from the after policy to
  the pre-DPO reference policy.
- Untouched-action retention: for each prompt, the relative
  ordering of the two held-out rejected actions (plus 2 versus
  plus 3). Neither logit is updated directly by DPO training
  on the train pairs, so any change in their ordering would
  be a side effect; the fraction of prompts whose ordering is
  preserved is reported.

Reward hacking probe: three named checks, run in code.
1. Constant policy probe: a policy that puts almost all mass
   on action 0 for every prompt. If the preference signal
   could be gamed by a fixed response, this policy would
   score high preference accuracy; it does not, because the
   chosen action varies by prompt.
2. Rejected-only suppression probe: take the pre-DPO logits
   and subtract 5.0 from only the train rejected action of
   each prompt. This games the train contrast directly
   without raising the chosen action at all. Its train,
   held-out, and task scores are recorded so the size of the
   gaming surface is measured, not asserted.
3. Trained policy check: the DPO policy is flagged as hacking
   or overfitting the train contrast if its held-out
   preference accuracy falls more than 0.2 below its train
   preference accuracy, or if its greedy task accuracy falls
   below the pre-DPO value. The outcome is recorded either way.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(os.path.dirname(_HERE), "level-2-build")
if _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

import grpo_build  # noqa: E402  Build module, reused unmodified

BETA = 0.1
LEARNING_RATE = 0.5
STEPS = 300


def log_probs(logits):
    probs = grpo_build.softmax_rows(np.asarray(logits, dtype=float))
    return np.log(probs + 1e-12)


def make_preference_pairs():
    """Deterministic pairs per the rule in the module docstring."""
    train_pairs = []
    held_out_pairs = []
    for p in range(grpo_build.N_PROMPTS):
        chosen = int(grpo_build.CORRECT[p])
        wrongs = sorted(
            [a for a in range(grpo_build.N_ACTIONS) if a != chosen],
            key=lambda a: (a - chosen) % grpo_build.N_ACTIONS)
        train_pairs.append((p, chosen, wrongs[0]))
        held_out_pairs.append((p, chosen, wrongs[1]))
        held_out_pairs.append((p, chosen, wrongs[2]))
    return train_pairs, held_out_pairs


def preference_metrics(logits, pairs):
    """Mean chosen-minus-rejected log-prob margin and accuracy."""
    lp = log_probs(logits)
    margins = [float(lp[p, c] - lp[p, r]) for p, c, r in pairs]
    correct = sum(1 for m in margins if m > 0)
    return {
        "n_pairs": len(pairs),
        "mean_margin": round(float(np.mean(margins)), 4),
        "preference_accuracy": round(correct / len(pairs), 4),
        "preference_correct": int(correct),
    }


def dpo_loss_and_grad(logits, ref_logits, pairs, beta=BETA):
    """Mean DPO loss and its hand-derived gradient.

    See the module docstring for the derivation. The gradient
    for one pair touches only that prompt's row and equals
    minus beta * (1 - sigmoid(beta * margin)) * (e_chosen -
    e_rejected), averaged over pairs.
    """
    logits = np.asarray(logits, dtype=float)
    lp = log_probs(logits)
    ref_lp = log_probs(ref_logits)
    grad = np.zeros_like(logits)
    total_loss = 0.0
    for p, c, r in pairs:
        margin = float((lp[p, c] - ref_lp[p, c])
                       - (lp[p, r] - ref_lp[p, r]))
        sig = 1.0 / (1.0 + np.exp(-beta * margin))
        total_loss += -float(np.log(sig + 1e-12))
        factor = -beta * (1.0 - sig) / len(pairs)
        grad[p, c] += factor
        grad[p, r] -= factor
    return float(total_loss / len(pairs)), grad


def train_dpo(steps=STEPS, lr=LEARNING_RATE, beta=BETA):
    """Full-batch DPO training from the Build initial policy."""
    train_pairs, held_out_pairs = make_preference_pairs()
    logits = grpo_build.init_policy()
    ref_logits = logits.copy()
    history = []
    loss = None
    for step in range(steps):
        loss, grad = dpo_loss_and_grad(logits, ref_logits,
                                       train_pairs, beta=beta)
        if step % 50 == 0 or step == steps - 1:
            history.append({
                "step": step,
                "loss": round(float(loss), 4),
                "train_mean_margin":
                    preference_metrics(logits, train_pairs)["mean_margin"],
                "held_out_mean_margin":
                    preference_metrics(logits, held_out_pairs)["mean_margin"],
            })
        logits = logits - lr * grad
    final_loss, _ = dpo_loss_and_grad(logits, ref_logits,
                                      train_pairs, beta=beta)
    return {
        "logits": logits,
        "ref_logits": ref_logits,
        "initial_logits": ref_logits.copy(),
        "final_loss": round(float(final_loss), 4),
        "history": history,
    }


def _untouched_order_retention(before_logits, after_logits):
    """Fraction of prompts keeping the plus2 versus plus3 order."""
    _, held_out_pairs = make_preference_pairs()
    kept = 0
    prompts = 0
    for p in range(grpo_build.N_PROMPTS):
        pair = [pr for pr in held_out_pairs if pr[0] == p]
        a2, a3 = pair[0][2], pair[1][2]
        before_order = before_logits[p, a2] > before_logits[p, a3]
        after_order = after_logits[p, a2] > after_logits[p, a3]
        kept += int(before_order == after_order)
        prompts += 1
    return round(kept / prompts, 4), int(kept), int(prompts)


def forgetting_checks(before_logits, after_logits):
    """Build GRPO task before and after, plus drift and retention."""
    before_eval = grpo_build.evaluate_policy(before_logits)
    after_eval = grpo_build.evaluate_policy(after_logits)
    ref_probs = grpo_build.softmax_rows(before_logits)
    after_probs = grpo_build.softmax_rows(after_logits)
    mean_kl = float(grpo_build.kl_divergence(after_probs, ref_probs).mean())
    retention, kept, prompts = _untouched_order_retention(
        before_logits, after_logits)
    return {
        "task": "Build GRPO checker task, 8 prompts, Build "
                "evaluate_policy, eval seed 41, 32 samples per prompt",
        "before": before_eval,
        "after": after_eval,
        "change": {
            "greedy_accuracy": round(
                after_eval["greedy_accuracy"]
                - before_eval["greedy_accuracy"], 4),
            "mean_reward": round(
                after_eval["mean_reward"] - before_eval["mean_reward"], 4),
            "greedy_correct": int(after_eval["greedy_correct"]
                                  - before_eval["greedy_correct"]),
        },
        "mean_kl_after_to_before": round(mean_kl, 4),
        "untouched_action_order_retention": retention,
        "untouched_action_order_kept": kept,
        "untouched_action_order_prompts": prompts,
    }


def reward_hacking_probe(before_logits, after_logits):
    """Three named checks from the module docstring, run in code."""
    train_pairs, held_out_pairs = make_preference_pairs()

    constant_logits = np.zeros_like(before_logits)
    constant_logits[:, 0] = 10.0
    constant_train = preference_metrics(constant_logits, train_pairs)
    constant_held = preference_metrics(constant_logits, held_out_pairs)
    constant_task = grpo_build.evaluate_policy(constant_logits)

    suppressed = np.array(before_logits, dtype=float).copy()
    for p, _c, r in train_pairs:
        suppressed[p, r] -= 5.0
    suppressed_train = preference_metrics(suppressed, train_pairs)
    suppressed_held = preference_metrics(suppressed, held_out_pairs)
    suppressed_task = grpo_build.evaluate_policy(suppressed)

    trained_train = preference_metrics(after_logits, train_pairs)
    trained_held = preference_metrics(after_logits, held_out_pairs)
    trained_task = grpo_build.evaluate_policy(after_logits)
    before_task = grpo_build.evaluate_policy(before_logits)
    flagged = bool(
        trained_held["preference_accuracy"]
        < trained_train["preference_accuracy"] - 0.2
        or trained_task["greedy_accuracy"] < before_task["greedy_accuracy"])

    return {
        "constant_policy_probe": {
            "description": "all mass on action 0 for every prompt",
            "train_preference_accuracy":
                constant_train["preference_accuracy"],
            "held_out_preference_accuracy":
                constant_held["preference_accuracy"],
            "greedy_accuracy": constant_task["greedy_accuracy"],
            "mean_reward": constant_task["mean_reward"],
        },
        "rejected_only_suppression_probe": {
            "description": "pre-DPO logits with 5.0 subtracted from "
                           "only the train rejected action per prompt, "
                           "chosen action never raised",
            "train_preference_accuracy":
                suppressed_train["preference_accuracy"],
            "held_out_preference_accuracy":
                suppressed_held["preference_accuracy"],
            "greedy_accuracy": suppressed_task["greedy_accuracy"],
            "mean_reward": suppressed_task["mean_reward"],
        },
        "trained_policy_check": {
            "description": "flag if held-out preference accuracy is "
                           "more than 0.2 below train, or greedy task "
                           "accuracy falls below the pre-DPO value",
            "train_preference_accuracy":
                trained_train["preference_accuracy"],
            "held_out_preference_accuracy":
                trained_held["preference_accuracy"],
            "greedy_accuracy": trained_task["greedy_accuracy"],
            "greedy_accuracy_before": before_task["greedy_accuracy"],
            "hacking_or_overfit_flagged": flagged,
        },
    }


def run_deep():
    """Run the full Deep experiment and return the results dict."""
    from memory_study import memory_study

    train_pairs, held_out_pairs = make_preference_pairs()
    initial = grpo_build.init_policy()
    before_train = preference_metrics(initial, train_pairs)
    before_held = preference_metrics(initial, held_out_pairs)
    before_task = grpo_build.evaluate_policy(initial)
    initial_loss, _ = dpo_loss_and_grad(initial, initial, train_pairs)

    trained = train_dpo()
    after_logits = trained["logits"]
    after_train = preference_metrics(after_logits, train_pairs)
    after_held = preference_metrics(after_logits, held_out_pairs)
    after_task = grpo_build.evaluate_policy(after_logits)

    forgetting = forgetting_checks(initial, after_logits)
    probe = reward_hacking_probe(initial, after_logits)

    return {
        "config": {
            "beta": BETA,
            "learning_rate": LEARNING_RATE,
            "steps": STEPS,
            "optimizer": "full-batch gradient descent",
            "policy": "tabular softmax, 8 prompts, 4 actions, "
                      "Build init_policy, policy seed 21",
            "reference_policy": "frozen copy of the initial policy",
            "train_pairs": len(train_pairs),
            "held_out_pairs": len(held_out_pairs),
            "pair_rule": "chosen is the checker-correct action; "
                         "train rejected is correct plus 1 modulo 4; "
                         "held-out rejected are correct plus 2 and "
                         "plus 3 modulo 4",
            "eval_seed": grpo_build.EVAL_SEED,
        },
        "dpo": {
            "loss_before": round(float(initial_loss), 4),
            "loss_after": trained["final_loss"],
            "train_pairs_before": before_train,
            "train_pairs_after": after_train,
            "held_out_pairs_before": before_held,
            "held_out_pairs_after": after_held,
            "task_before": before_task,
            "task_after": after_task,
            "history": trained["history"],
        },
        "forgetting_checks": forgetting,
        "reward_hacking_probe": probe,
        "qlora_memory_study": memory_study(),
    }
