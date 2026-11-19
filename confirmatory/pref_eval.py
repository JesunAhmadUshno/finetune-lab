"""Held-out preference accuracy on a seed's test pairs.

Accuracy: fraction of pairs where the policy's summed completion
log-probability of chosen exceeds that of rejected (RUN-PLAN.md).
Usage: pref_eval.py --seed N --model <path> --tag <name>
Writes runs/evals/<tag>.summary.json with the per-pair wins in
runs/evals/<tag>.jsonl.
"""
import argparse
import os

import torch

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    timer = common.Timer()
    tok = common.load_tokenizer()
    pairs = common.read_jsonl(os.path.join(
        common.RUNS, "pairs", f"seed{args.seed}", "test_pairs.jsonl"))
    model = common.load_policy(args.model, dtype=torch.bfloat16)
    prompts = [p["prompt"] for p in pairs]
    chosen = [p["chosen"] for p in pairs]
    rejected = [p["rejected"] for p in pairs]
    lp_c = common.batch_logps(model, tok, prompts, chosen)
    lp_r = common.batch_logps(model, tok, prompts, rejected)
    rows = [{"win": c > r, "lp_chosen": c, "lp_rejected": r}
            for c, r in zip(lp_c, lp_r)]
    wins = sum(r["win"] for r in rows)
    outdir = os.path.join(common.RUNS, "evals")
    common.write_jsonl(os.path.join(outdir, args.tag + ".jsonl"), rows)
    summary = {"tag": args.tag, "seed": args.seed, "model": args.model,
               "n_pairs": len(rows), "wins": wins,
               "preference_accuracy": wins / len(rows),
               "gpu_hours": timer.hours()}
    common.dump_json(os.path.join(outdir, args.tag + ".summary.json"), summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
