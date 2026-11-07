"""Run the rigor study and write rigor/results_rigor.json.

Deterministic: fixed seeds, no timing fields. Run it twice and the
file is byte identical. Takes well under a minute on one CPU core.
"""
import json
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rigor_study import run_all  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    results = {
        "project": "P10 finetune-lab",
        "level": "rigor",
        "environment": f"CPU, Python {platform.python_version()}, NumPy {np.__version__}",
        "note": "Synthetic NumPy teaching models only, rerun across seeds and settings. No real LLM, no real text data, no GPU run.",
        **run_all(),
    }
    out = os.path.join(HERE, "results_rigor.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    s = results["seeds"]
    for track, key in (("sft", "final_test_accuracy"), ("grpo", "final_mean_reward"),
                       ("dpo", "held_out_margin")):
        print(track, key, s[track]["summary"][key])
    print("wrote", out)


if __name__ == "__main__":
    main()
