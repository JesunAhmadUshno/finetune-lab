"""Run Level 3, Deep for finetune-lab and write results_deep.json.

Deterministic: fixed seeds everywhere, no timing fields. Run it
twice and results_deep.json is byte identical.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dpo_deep import run_deep

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    deep = run_deep()
    results = {
        "level": "3-deep",
        "project": "P10 finetune-lab",
        "date": "2026-11-05",
        "environment": "Muse VM, CPU, Python 3.12, NumPy 1.26.4",
        "note": "Synthetic NumPy teaching policy only. DPO on a "
                "tabular toy policy with synthetic preferences. "
                "QLoRA study is accounting from parameter counts, "
                "not measured GPU memory. No real LLM, no real "
                "text data, no GPU run.",
        "deep": deep,
    }
    out = os.path.join(HERE, "results_deep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    d = deep["dpo"]
    print("DPO train margin: before", d["train_pairs_before"]["mean_margin"],
          "after", d["train_pairs_after"]["mean_margin"])
    print("DPO held-out margin: before", d["held_out_pairs_before"]["mean_margin"],
          "after", d["held_out_pairs_after"]["mean_margin"])
    print("DPO held-out preference accuracy: before",
          d["held_out_pairs_before"]["preference_accuracy"],
          "after", d["held_out_pairs_after"]["preference_accuracy"])
    print("Task greedy accuracy: before", d["task_before"]["greedy_accuracy"],
          "after", d["task_after"]["greedy_accuracy"])
    print("wrote", out)


if __name__ == "__main__":
    main()
