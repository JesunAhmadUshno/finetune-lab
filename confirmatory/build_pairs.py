"""Build DPO preference pairs for one seed (RUN-PLAN.md).

Prompts: GSM8K train in dataset order, first 2,000 (training
pairs) and all 1,319 test prompts (held-out measurement pairs).
Chosen: the gold solution, calculator annotations stripped.
Rejected: one completion sampled from the seed's full fine-tune
SFT policy at temperature 0.9 whose extracted answer fails the
checker (4 samples per round, up to 3 rounds; a prompt with no
failing sample is skipped and counted).
"""
import argparse
import os

import torch

import common

SMOKE = os.environ.get("CONF_SMOKE") == "1"


def build(model, tok, rows, limit):
    rows = rows[:limit]
    prompts = [common.gsm8k_prompt(r["question"]) for r in rows]
    golds = [common.gold_answer(r["answer"]) for r in rows]
    rejected = [None] * len(rows)
    pending = list(range(len(rows)))
    for _round in range(3):
        if not pending:
            break
        comps = common.generate_batch(
            model, tok, [prompts[i] for i in pending],
            max_new_tokens=320, batch_size=32, do_sample=True,
            temperature=0.9, num_return_sequences=4)
        nxt = []
        for k, i in enumerate(pending):
            mine = comps[k * 4:(k + 1) * 4]
            pick = None
            for c in mine:
                if not common.answers_equal(common.extract_answer(c),
                                             golds[i]):
                    pick = c
                    break
            if pick is None:
                nxt.append(i)
            else:
                rejected[i] = pick
        pending = nxt
    pairs, skipped = [], 0
    for i, r in enumerate(rows):
        if rejected[i] is None:
            skipped += 1
            continue
        chosen = common.strip_calc_annotations(r["answer"])
        pairs.append({"prompt": prompts[i], "chosen": chosen,
                      "rejected": rejected[i], "gold": golds[i]})
    return pairs, skipped, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    timer = common.Timer()
    tok = common.load_tokenizer()
    sft = os.path.join(common.RUNS, "h1", "full", f"seed{args.seed}", "ckpt")
    model = common.load_policy(sft, dtype=torch.bfloat16)
    outdir = os.path.join(common.RUNS, "pairs", f"seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)

    train_rows = common.read_jsonl(os.path.join(common.DATA, "gsm8k_train.jsonl"))
    test_rows = common.read_jsonl(os.path.join(common.DATA, "gsm8k_test.jsonl"))
    n_train = 100 if SMOKE else 2000
    n_test = 100 if SMOKE else len(test_rows)
    tp, tskip, tatt = build(model, tok, train_rows, n_train)
    common.write_jsonl(os.path.join(outdir, "train_pairs.jsonl"), tp)
    sp, sskip, satt = build(model, tok, test_rows, n_test)
    common.write_jsonl(os.path.join(outdir, "test_pairs.jsonl"), sp)
    common.dump_json(os.path.join(outdir, "counts.json"), {
        "seed": args.seed,
        "train": {"attempted": tatt, "pairs": len(tp), "skipped": tskip},
        "test": {"attempted": satt, "pairs": len(sp), "skipped": sskip},
        "gpu_hours": timer.hours(), "smoke": SMOKE})
    print("pairs done", args.seed, len(tp), len(sp), flush=True)


if __name__ == "__main__":
    main()
