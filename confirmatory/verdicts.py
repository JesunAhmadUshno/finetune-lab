"""Phase 3: verdicts under the 0006 lock, recomputed from raw
per-seed outputs (never copied from run logs or summaries).

Reads runs/evals/*.jsonl (per-item records) and runs/grpo/**
metrics.json (per-seed final measurements), recomputes every
accuracy and mean, applies the lock's verdict rule with bootstrap
95 percent intervals (10,000 resamples over seeds), and writes
results/<out-name>.json. Each arm is computed independently: an
arm whose raw files are missing is recorded as not computed,
with the reason, and never blocks the other arms.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results")
E = os.path.join(RUNS, "evals")
RNG = np.random.default_rng(20261117)


def boot_ci(values, n=10000):
    v = np.asarray(values, dtype=float)
    idx = RNG.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def accuracy_from_jsonl(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return sum(r["correct"] for r in rows) / len(rows), len(rows)


def pref_from_jsonl(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return sum(r["win"] for r in rows) / len(rows), len(rows)


def verdict_at_least(mean, ci, line):
    if mean >= line and ci[0] >= line:
        return "CONFIRMED"
    if mean < line and ci[1] < line:
        return "REFUTED"
    return "INCONCLUSIVE"


def verdict_at_most(mean, ci, line):
    if mean <= line and ci[1] <= line:
        return "CONFIRMED"
    if mean > line and ci[0] > line:
        return "REFUTED"
    return "INCONCLUSIVE"


def arm_h1(seeds):
    gaps, full_accs, lora_accs = [], [], []
    for s in seeds:
        fa, _ = accuracy_from_jsonl(os.path.join(E, f"h1_full_seed{s}.jsonl"))
        la, _ = accuracy_from_jsonl(os.path.join(E, f"h1_lora_seed{s}.jsonl"))
        full_accs.append(fa)
        lora_accs.append(la)
        gaps.append(fa - la)
    m, ci = float(np.mean(gaps)), boot_ci(gaps)
    return {"full_mean": float(np.mean(full_accs)),
            "lora_mean": float(np.mean(lora_accs)),
            "per_seed_gap": gaps, "gap_mean": m, "gap_ci95": ci,
            "line": 0.05, "verdict": verdict_at_most(m, ci, 0.05)}


def arm_h3(seeds):
    gains, sft_accs, dpo_accs = [], [], []
    for s in seeds:
        ps, _ = pref_from_jsonl(os.path.join(E, f"pref_sft_seed{s}.jsonl"))
        pd_, _ = pref_from_jsonl(os.path.join(E, f"pref_dpo_seed{s}.jsonl"))
        sft_accs.append(ps)
        dpo_accs.append(pd_)
        gains.append(pd_ - ps)
    m, ci = float(np.mean(gains)), boot_ci(gains)
    return {"sft_pref_mean": float(np.mean(sft_accs)),
            "dpo_pref_mean": float(np.mean(dpo_accs)),
            "per_seed_gain": gains, "gain_mean": m, "gain_ci95": ci,
            "line": 0.25, "verdict": verdict_at_least(m, ci, 0.25)}


def arm_h4(seeds):
    drops, pre_accs, post_accs = [], [], []
    for s in seeds:
        pre, _ = accuracy_from_jsonl(os.path.join(E, f"h1_full_seed{s}.jsonl"))
        post, _ = accuracy_from_jsonl(os.path.join(E, f"h4_mis_seed{s}.jsonl"))
        pre_accs.append(pre)
        post_accs.append(post)
        drops.append(pre - post)
    m, ci = float(np.mean(drops)), boot_ci(drops)
    return {"pre_mean": float(np.mean(pre_accs)),
            "post_mean": float(np.mean(post_accs)),
            "per_seed_drop": drops, "drop_mean": m, "drop_ci95": ci,
            "line": 0.15, "verdict": verdict_at_least(m, ci, 0.15)}


def arm_h2(seeds):
    h2 = {}
    for b in ("0.0", "0.04", "0.1"):
        rs, ks = [], []
        for s in seeds:
            with open(os.path.join(RUNS, "grpo", f"beta{b}",
                                   f"seed{s}", "metrics.json")) as f:
                mj = json.load(f)
            rs.append(mj["final_reward"])
            ks.append(mj["final_kl"])
        h2[b] = {"reward_mean": float(np.mean(rs)),
                 "kl_mean": float(np.mean(ks)),
                 "reward_per_seed": rs, "kl_per_seed": ks}
    d_r = [h2["0.1"]["reward_per_seed"][i] - h2["0.0"]["reward_per_seed"][i]
           for i in range(len(seeds))]
    d_k = [h2["0.1"]["kl_per_seed"][i] - h2["0.0"]["kl_per_seed"][i]
           for i in range(len(seeds))]
    ci_r, ci_k = boot_ci(d_r), boot_ci(d_k)
    mr, mk = float(np.mean(d_r)), float(np.mean(d_k))
    if mr < 0 and mk < 0 and ci_r[1] < 0 and ci_k[1] < 0:
        v = "CONFIRMED"
    elif (mr > 0 and ci_r[0] > 0) or (mk > 0 and ci_k[0] > 0):
        v = "REFUTED"
    else:
        v = "INCONCLUSIVE"
    h2["diff_reward_0.1_minus_0"] = {"mean": mr, "ci95": ci_r}
    h2["diff_kl_0.1_minus_0"] = {"mean": mk, "ci95": ci_k}
    h2["verdict"] = v
    return h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out-name", default="confirmatory-results")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    res = {"seeds": seeds, "n_seeds": len(seeds),
           "complete_seed_set": len(seeds) == 10}
    try:
        base_acc, _ = accuracy_from_jsonl(os.path.join(E, "base.jsonl"))
        res["base_gsm8k_accuracy"] = base_acc
    except Exception as exc:
        res["base_gsm8k_accuracy"] = {"status": "not computed",
                                      "reason": str(exc)}
    for name, fn in (("H1", arm_h1), ("H3", arm_h3), ("H4", arm_h4),
                     ("H2", arm_h2)):
        try:
            res[name] = fn(seeds)
        except Exception as exc:
            res[name] = {"status": "not computed", "reason": str(exc)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, args.out_name + ".json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
