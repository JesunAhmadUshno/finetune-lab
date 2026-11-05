"""Run Level 2, Build for finetune-lab and write results_build.json.

Deterministic: fixed seeds everywhere, no timing fields. Run it
twice and results_build.json is byte identical. The checkpoint
file adapter_checkpoint.npz (plus its JSON sidecar) is written
next to this script and reloaded for the round-trip check; the
checkpoint bytes are an output artifact, not part of the JSON.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpo_build import run_grpo
from pipeline import run_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ckpt = os.path.join(HERE, "adapter_checkpoint.npz")
    sft = run_pipeline(checkpoint_path=ckpt)
    grpo = run_grpo()
    results = {
        "level": "2-build",
        "project": "P10 finetune-lab",
        "date": "2026-11-05",
        "environment": "Muse VM, CPU, Python 3.12, NumPy 1.26.4",
        "note": "Synthetic tasks and NumPy teaching models only. SFT track is a LoRA pipeline on synthetic vectors. GRPO track is a tabular toy policy on a synthetic checker task. No real LLM, no real text data, no GPU run.",
        "sft_pipeline": sft,
        "grpo_track": grpo,
    }
    out = os.path.join(HERE, "results_build.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    bt = sft["before_training"]["test"]
    at = sft["after_training"]["test"]
    print("SFT test accuracy: before", bt["accuracy"], "after", at["accuracy"])
    print("SFT test loss: before", bt["loss"], "after", at["loss"])
    print("GRPO eval reward: before", grpo["before_training"]["mean_reward"],
          "after", grpo["after_training"]["mean_reward"])
    print("GRPO greedy accuracy: before", grpo["before_training"]["greedy_accuracy"],
          "after", grpo["after_training"]["greedy_accuracy"])
    print("wrote", out)


if __name__ == "__main__":
    main()
