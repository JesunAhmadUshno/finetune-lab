"""Rigor study for finetune-lab: seeds, sweeps, baselines, ablations.

Every level result in this repo is one seed configuration. This
module reruns the same experiments with the seeds and settings made
parameters, so the single numbers can be read with their spread:

1. Multi-seed: SFT (LoRA rank 2), GRPO and DPO over 10 seed
   offsets. Offset 0 is the recorded configuration and reproduces
   the level results files exactly (tests/test_rigor.py checks it).
   Offset s adds 1000 * s to every seed of that run (data, base,
   target, adapter, shuffle for SFT; policy, train, eval for GRPO;
   policy and eval for DPO). Summaries give the mean, the sample
   standard deviation and a percentile bootstrap 95% interval of
   the mean (10,000 resamples, fixed bootstrap seed).
2. SFT baselines and selection: full fine-tuning of all 64 weights
   against LoRA ranks 1 to 4 on the same splits, plus best-validation
   epoch selection (early stopping) against the final epoch.
3. GRPO budget and KL beta sweep: eval reward, greedy accuracy and
   KL to the reference at iteration checkpoints, for beta in
   {0, 0.01, 0.04, 0.1}.
4. GRPO ablations: no clipping, no KL term, and a plain REINFORCE
   update with the same normalized group baseline (one inner epoch,
   no clip, no KL). These are tabular stand-ins for the GRPO against
   PPO question, not a PPO implementation (there is no critic).
5. DPO sweep over beta, learning rate and steps on the recorded
   seed (DPO training itself is deterministic given the init).
6. Misaligned forgetting check: DPO where the chosen action
   contradicts the checker on the first k prompts.

All synthetic, NumPy, CPU. No real model, no text.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("level-2-build", "level-3-deep"):
    path = os.path.join(_ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import dpo_deep  # noqa: E402
import grpo_build  # noqa: E402
import pipeline  # noqa: E402

N_SEEDS = 10
SEED_STRIDE = 1000
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 12345


def summarize(values):
    """Mean, sample std, min, max and a bootstrap 95% CI of the mean."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(v), size=(BOOTSTRAP_RESAMPLES, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(v.mean()), 4),
            "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(len(v))}


# ---------------------------------------------------------------- SFT

def trainable_parameters(rank):
    """'full' is the whole 16 x 4 layer; an integer is a LoRA rank."""
    if rank == "full":
        return pipeline.IN_DIM * pipeline.NUM_CLASSES
    return pipeline.lora_parameter_count(rank=rank)


def make_dataset(seed_offset=0):
    """pipeline.make_dataset with every seed shifted by the offset."""
    k = SEED_STRIDE * seed_offset
    rng = np.random.default_rng(pipeline.DATA_SEED + k)
    total = pipeline.N_TRAIN + pipeline.N_VAL + pipeline.N_TEST
    x = rng.standard_normal((total, pipeline.IN_DIM))
    w_base = np.random.default_rng(pipeline.BASE_SEED + k).standard_normal(
        (pipeline.IN_DIM, pipeline.NUM_CLASSES)) * 0.5
    rt = np.random.default_rng(pipeline.TARGET_SEED + k)
    a_true = rt.standard_normal((pipeline.IN_DIM, pipeline.RANK)) * 0.6
    b_true = rt.standard_normal((pipeline.RANK, pipeline.NUM_CLASSES)) * 0.6
    y = np.argmax(x @ (w_base + pipeline.SCALING * (a_true @ b_true)), axis=1)
    n1, n2 = pipeline.N_TRAIN, pipeline.N_TRAIN + pipeline.N_VAL
    return {"x_train": x[:n1], "y_train": y[:n1], "x_val": x[n1:n2],
            "y_val": y[n1:n2], "x_test": x[n2:], "y_test": y[n2:], "w_base": w_base}


def _score(x, y, w):
    probs = pipeline.softmax(x @ w)
    loss = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())
    return round(float((probs.argmax(axis=1) == y).mean()), 4), round(loss, 4)


def run_sft(seed_offset=0, rank=pipeline.RANK, epochs=pipeline.EPOCHS,
            lr=pipeline.LEARNING_RATE, batch_size=pipeline.BATCH_SIZE):
    """Minibatch SFT as in pipeline.train_sft, with per-epoch scoring.

    rank='full' trains the whole weight matrix from the base instead
    of an adapter. Best-val selection picks the epoch with the lowest
    validation loss (ties: earliest), then reports test at that epoch.
    """
    k = SEED_STRIDE * seed_offset
    ds = make_dataset(seed_offset)
    w_base = ds["w_base"]
    full = rank == "full"
    if full:
        w = w_base.copy()
    else:
        scaling = pipeline.ALPHA / rank
        a = np.random.default_rng(pipeline.ADAPTER_SEED + k).standard_normal(
            (pipeline.IN_DIM, rank)) * 0.1
        b = np.zeros((rank, pipeline.NUM_CLASSES))

    def weights():
        return w if full else w_base + scaling * (a @ b)

    before_test, _ = _score(ds["x_test"], ds["y_test"], weights())
    x_tr, y_tr = ds["x_train"], ds["y_train"]
    per_epoch = []
    for epoch in range(epochs):
        for idx in pipeline.batch_indices(len(y_tr), batch_size, epoch=epoch,
                                          shuffle_seed=pipeline.SHUFFLE_SEED + k):
            xb, yb = x_tr[idx], y_tr[idx]
            d = pipeline.softmax(xb @ weights())
            d[np.arange(len(yb)), yb] -= 1.0
            d_w = xb.T @ (d / len(yb))
            if full:
                w = w - lr * d_w
            else:
                d_delta = scaling * d_w
                a, b = a - lr * (d_delta @ b.T), b - lr * (a.T @ d_delta)
        val_acc, val_loss = _score(ds["x_val"], ds["y_val"], weights())
        test_acc, _ = _score(ds["x_test"], ds["y_test"], weights())
        per_epoch.append((val_loss, epoch + 1, val_acc, test_acc))
    best = min(per_epoch)
    return {"seed_offset": seed_offset, "rank": rank,
            "trainable_parameters": trainable_parameters(rank),
            "before_test_accuracy": before_test,
            "final_test_accuracy": per_epoch[-1][3],
            "final_val_accuracy": per_epoch[-1][2],
            "best_val_epoch": best[1], "best_val_test_accuracy": best[3],
            "peak_val_accuracy": max(p[2] for p in per_epoch)}


# --------------------------------------------------------------- GRPO

def _init_policy(seed):
    return np.random.default_rng(seed).standard_normal(
        (grpo_build.N_PROMPTS, grpo_build.N_ACTIONS)) * 0.1


def run_grpo_cfg(seed_offset=0, beta=grpo_build.BETA_KL, eps=grpo_build.EPS_CLIP,
                 group_size=grpo_build.GROUP_SIZE, iterations=grpo_build.ITERATIONS,
                 inner_epochs=2, lr=grpo_build.LEARNING_RATE, checkpoints=()):
    """grpo_build.run_grpo with seeds and knobs as parameters.

    Checkpoint evaluations use the run's fixed eval seed so every
    checkpoint is scored on the same evaluation stream.
    """
    k = SEED_STRIDE * seed_offset
    eval_seed = grpo_build.EVAL_SEED + k
    logits = _init_policy(grpo_build.POLICY_SEED + k)
    ref = grpo_build.softmax_rows(logits.copy())
    rng = np.random.default_rng(grpo_build.TRAIN_SEED + k)
    marks = []
    for it in range(iterations):
        probs = grpo_build.softmax_rows(logits)
        samples = []
        for p in range(grpo_build.N_PROMPTS):
            actions = rng.choice(grpo_build.N_ACTIONS, size=group_size, p=probs[p])
            rewards = np.array([grpo_build.checker(p, act) for act in actions])
            adv, _, _ = grpo_build.group_advantages(rewards)
            samples.append((actions, adv, probs[p].copy()))
        for _ in range(inner_epochs):
            _, grad = grpo_build.grpo_loss_and_grad(logits, ref, samples, beta=beta, eps=eps)
            logits = logits - lr * grad
        if it + 1 in checkpoints:
            ev = grpo_build.evaluate_policy(logits, seed=eval_seed)
            marks.append({"iteration": it + 1, "mean_reward": ev["mean_reward"],
                          "greedy_accuracy": ev["greedy_accuracy"],
                          "kl_to_ref": round(float(grpo_build.kl_divergence(
                              grpo_build.softmax_rows(logits), ref).mean()), 4)})
    ev = grpo_build.evaluate_policy(logits, seed=eval_seed)
    return {"seed_offset": seed_offset, "final_mean_reward": ev["mean_reward"],
            "final_greedy_accuracy": ev["greedy_accuracy"],
            "final_kl_to_ref": round(float(grpo_build.kl_divergence(
                grpo_build.softmax_rows(logits), ref).mean()), 4),
            "checkpoints": marks}


# ---------------------------------------------------------------- DPO

def misaligned_pairs(k):
    """Train pairs where the first k prompts prefer a wrong action.

    On prompt p < k the chosen action is correct plus 1 modulo 4 and
    the rejected action is the checker-correct one, so DPO pulls
    against the task there. Prompts k..7 keep the normal train pair.
    """
    pairs = []
    for p in range(grpo_build.N_PROMPTS):
        c = int(grpo_build.CORRECT[p])
        w = (c + 1) % grpo_build.N_ACTIONS
        pairs.append((p, w, c) if p < k else (p, c, w))
    return pairs


def run_dpo_cfg(seed_offset=0, beta=dpo_deep.BETA, lr=dpo_deep.LEARNING_RATE,
                steps=dpo_deep.STEPS, train_pairs=None):
    k = SEED_STRIDE * seed_offset
    init = _init_policy(grpo_build.POLICY_SEED + k)
    std_train, held = dpo_deep.make_preference_pairs()
    pairs = train_pairs or std_train
    logits = init.copy()
    for _ in range(steps):
        _, grad = dpo_deep.dpo_loss_and_grad(logits, init, pairs, beta=beta)
        logits = logits - lr * grad
    ev = grpo_build.evaluate_policy(logits, seed=grpo_build.EVAL_SEED + k)
    greedy_ok = grpo_build.softmax_rows(logits).argmax(axis=1) == grpo_build.CORRECT
    return {"seed_offset": seed_offset,
            "train_margin": dpo_deep.preference_metrics(logits, std_train)["mean_margin"],
            "held_out_margin": dpo_deep.preference_metrics(logits, held)["mean_margin"],
            "held_out_accuracy": dpo_deep.preference_metrics(logits, held)["preference_accuracy"],
            "greedy_accuracy": ev["greedy_accuracy"], "mean_reward": ev["mean_reward"],
            "greedy_correct_by_prompt": [bool(g) for g in greedy_ok],
            "kl_to_init": round(float(grpo_build.kl_divergence(
                grpo_build.softmax_rows(logits), grpo_build.softmax_rows(init)).mean()), 4)}


# ---------------------------------------------------------------- run

def _seed_track(fn, keys, **kw):
    rows = [fn(seed_offset=s, **kw) for s in range(N_SEEDS)]
    return {"per_seed": [{k: r[k] for k in ("seed_offset",) + keys} for r in rows],
            "summary": {k: summarize([r[k] for r in rows]) for k in keys}}


def run_all():
    sft_keys = ("before_test_accuracy", "final_test_accuracy",
                "best_val_test_accuracy", "best_val_epoch")
    grpo_keys = ("final_mean_reward", "final_greedy_accuracy", "final_kl_to_ref")
    dpo_keys = ("train_margin", "held_out_margin", "held_out_accuracy",
                "greedy_accuracy", "mean_reward", "kl_to_init")

    seeds = {"n_seeds": N_SEEDS,
             "seed_rule": "seed offset s adds 1000 * s to every seed of the run; s = 0 is the recorded configuration",
             "sft": _seed_track(run_sft, sft_keys),
             "grpo": _seed_track(run_grpo_cfg, grpo_keys),
             "dpo": _seed_track(run_dpo_cfg, dpo_keys)}

    sft_baselines = {"note": "same splits and minibatch schedule; LoRA keeps alpha 4 so scaling is 4 / rank; the target change is rank 2 by construction",
                     "rows": []}
    for rank in ("full", 1, 2, 3, 4):
        t = _seed_track(run_sft, ("final_test_accuracy", "best_val_test_accuracy"), rank=rank)
        sft_baselines["rows"].append({"rank": rank, "trainable_parameters": trainable_parameters(rank),
                                      "final_test_accuracy": t["summary"]["final_test_accuracy"],
                                      "best_val_test_accuracy": t["summary"]["best_val_test_accuracy"]})

    marks = (5, 10, 20, 30, 40, 60)
    budget = {"checkpoints": list(marks),
              "cost_unit": "sampled actions per iteration = 8 prompts x group size 8 = 64",
              "rows": []}
    for beta in (0.0, 0.01, 0.04, 0.1):
        runs = [run_grpo_cfg(seed_offset=s, beta=beta, checkpoints=marks) for s in range(N_SEEDS)]
        budget["rows"].append({"beta": beta, "checkpoints": [
            {"iteration": m, "samples": m * 64,
             **{key: summarize([r["checkpoints"][i][key] for r in runs])
                for key in ("mean_reward", "greedy_accuracy", "kl_to_ref")}}
            for i, m in enumerate(marks)]})

    variants = [("GRPO (clip 0.2, KL beta 0.04, 2 inner epochs)", {}),
                ("GRPO without clipping", {"eps": 1e9}),
                ("GRPO without KL term", {"beta": 0.0}),
                ("REINFORCE with group baseline (1 epoch, no clip, no KL)",
                 {"eps": 1e9, "beta": 0.0, "inner_epochs": 1})]
    ablations = {"note": "tabular stand-in for the GRPO against PPO comparison; no critic is trained", "rows": []}
    for name, kw in variants:
        t = _seed_track(run_grpo_cfg, grpo_keys, **kw)
        ablations["rows"].append({"variant": name, **t["summary"]})

    grid = {"beta": [0.05, 0.1, 0.2, 0.5], "learning_rate": [0.1, 0.5, 1.0], "steps": [100, 300, 1000]}
    sweep = {"grid": grid, "seed_offset": 0, "rows": []}
    for beta in grid["beta"]:
        for lr in grid["learning_rate"]:
            for steps in grid["steps"]:
                r = run_dpo_cfg(beta=beta, lr=lr, steps=steps)
                sweep["rows"].append({"beta": beta, "learning_rate": lr, "steps": steps,
                                      **{k: r[k] for k in dpo_keys}})

    misaligned = {"rule": "on the first k prompts the chosen action is correct plus 1 and the rejected action is the correct one", "rows": []}
    for k in (0, 2, 4):
        runs = [run_dpo_cfg(seed_offset=s, train_pairs=misaligned_pairs(k)) for s in range(N_SEEDS)]
        mis = [np.mean(r["greedy_correct_by_prompt"][:k]) if k else None for r in runs]
        ali = [np.mean(r["greedy_correct_by_prompt"][k:]) for r in runs]
        misaligned["rows"].append({
            "k_misaligned": k,
            "greedy_accuracy": summarize([r["greedy_accuracy"] for r in runs]),
            "mean_reward": summarize([r["mean_reward"] for r in runs]),
            "greedy_on_misaligned_prompts": summarize(mis) if k else None,
            "greedy_on_aligned_prompts": summarize(ali)})
    before = [grpo_build.evaluate_policy(_init_policy(grpo_build.POLICY_SEED + SEED_STRIDE * s),
                                         seed=grpo_build.EVAL_SEED + SEED_STRIDE * s)
              for s in range(N_SEEDS)]
    misaligned["before_greedy_accuracy"] = summarize([b["greedy_accuracy"] for b in before])

    return {"seeds": seeds, "sft_baselines": sft_baselines, "grpo_beta_budget": budget,
            "grpo_ablations": ablations, "dpo_sweep": sweep, "misaligned_forgetting": misaligned,
            "site_fixtures": {"init_policy_logits": grpo_build.init_policy().tolist(),
                              "note": "Build init_policy (seed 21) logits, so the site's DPO port starts from the recorded policy"}}
