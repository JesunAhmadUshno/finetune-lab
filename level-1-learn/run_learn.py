"""Run Level 1, Learn for finetune-lab and write results_learn.json.

Deterministic: fixed seeds, no timing fields, no randomness beyond the
seeded generators in lora_learn. Run it twice and the file is identical.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpo_learn import run_grpo_examples
from lora_learn import run_lora

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    lora = run_lora()
    results = {
        "level": "1-learn",
        "project": "P10 finetune-lab",
        "date": "2026-11-05",
        "environment": "Muse VM, CPU, Python 3.12, NumPy 1.26.4",
        "note": "Synthetic task and NumPy teaching model only. No real LLM, no GPU run, no SFT or RL policy run at this level.",
        "lora": lora,
        "grpo_examples": run_grpo_examples(),
    }
    out = os.path.join(HERE, "results_learn.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    before = lora["before_training"]
    after = lora["after_training"]
    print("LoRA test accuracy: before", before["test_accuracy"],
          "after", after["test_accuracy"])
    print("LoRA trainable parameters:", lora["lora"]["trainable_parameters"],
          "of full", lora["lora"]["full_finetune_parameters"])
    print("GRPO example groups:", len(results["grpo_examples"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
