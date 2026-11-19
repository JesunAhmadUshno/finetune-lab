"""Build the frozen data slices for the confirmatory run.

Alpaca: shuffled with seed 0; train rows 0..9999, validation the
next 1,000, held-out the next 1,000 (lock: docs/decisions/0006).
GSM8K: train and test as published. Outputs land in data/ as
jsonl with a manifest of counts and hashes.
"""
import hashlib
import json
import os
import random

from datasets import load_dataset

import common


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    os.makedirs(common.DATA, exist_ok=True)
    alpaca = load_dataset("tatsu-lab/alpaca")["train"]
    assert len(alpaca) == 52002, len(alpaca)
    idx = list(range(len(alpaca)))
    random.Random(0).shuffle(idx)
    slices = {
        "alpaca_train": idx[:10000],
        "alpaca_val": idx[10000:11000],
        "alpaca_heldout": idx[11000:12000],
    }
    manifest = {}
    for name, rows in slices.items():
        out = []
        for i in rows:
            r = alpaca[i]
            out.append({"instruction": r["instruction"],
                        "input": r["input"], "output": r["output"]})
        path = os.path.join(common.DATA, name + ".jsonl")
        common.write_jsonl(path, out)
        manifest[name] = {"n": len(out), "sha256": sha(path)}

    gsm = load_dataset("openai/gsm8k", "main")
    for split in ("train", "test"):
        out = [{"question": r["question"], "answer": r["answer"]}
               for r in gsm[split]]
        path = os.path.join(common.DATA, f"gsm8k_{split}.jsonl")
        common.write_jsonl(path, out)
        manifest[f"gsm8k_{split}"] = {"n": len(out), "sha256": sha(path)}

    common.dump_json(os.path.join(common.DATA, "manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
