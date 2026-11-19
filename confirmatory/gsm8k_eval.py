"""GSM8K test exact-match evaluation under the one fixed protocol
(RUN-PLAN.md): zero-shot Alpaca-format prompt, greedy decoding,
extraction after the last #### or the last number.

Usage: gsm8k_eval.py --model <path-or-base> --tag <name>
Writes runs/evals/<tag>.jsonl (per item) and <tag>.summary.json.
"""
import argparse
import os

import common

SMOKE = os.environ.get("CONF_SMOKE") == "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    timer = common.Timer()
    tok = common.load_tokenizer()
    rows = common.read_jsonl(os.path.join(common.DATA, "gsm8k_test.jsonl"))
    if SMOKE:
        rows = rows[:100]
    model = common.load_policy(args.model, dtype=__import__("torch").bfloat16)
    prompts = [common.gsm8k_prompt(r["question"]) for r in rows]
    completions = common.generate_batch(model, tok, prompts,
                                        max_new_tokens=320, batch_size=64)
    out = []
    correct = 0
    for r, comp in zip(rows, completions):
        gold = common.gold_answer(r["answer"])
        pred = common.extract_answer(comp)
        ok = common.answers_equal(pred, gold)
        correct += ok
        out.append({"question": r["question"], "gold": gold,
                    "pred": pred, "correct": ok, "completion": comp})
    outdir = os.path.join(common.RUNS, "evals")
    common.write_jsonl(os.path.join(outdir, args.tag + ".jsonl"), out)
    summary = {"tag": args.tag, "model": args.model or common.MODEL_ID,
               "n": len(out), "correct": correct,
               "accuracy": correct / len(out),
               "gpu_hours": timer.hours(), "smoke": SMOKE}
    common.dump_json(os.path.join(outdir, args.tag + ".summary.json"), summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
