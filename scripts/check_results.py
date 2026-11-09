"""Rerun every runner and check that the committed results are unchanged.

Each runner rewrites its results JSON next to itself. This script keeps
the committed bytes, runs the four runners, compares, and puts the
committed bytes back, so the working tree is never left changed.

Ignored keys: "environment" (Python and NumPy version string) and the two
Build adapter checkpoint hashes "a_hash" and "b_hash", which change with
platform and NumPy build while every rounded metric stays the same
(REPORT.md, section 6). Everything else must match exactly.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("level-1-learn/run_learn.py", "level-1-learn/results_learn.json"),
        ("level-2-build/run_build.py", "level-2-build/results_build.json"),
        ("level-3-deep/run_deep.py", "level-3-deep/results_deep.json"),
        ("rigor/run_rigor.py", "rigor/results_rigor.json")]
VOLATILE = {"environment", "a_hash", "b_hash"}


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def main():
    failed = []
    for runner, out in RUNS:
        path = os.path.join(ROOT, out)
        committed = open(path, "rb").read()
        try:
            subprocess.run([sys.executable, os.path.join(ROOT, runner)], check=True, stdout=subprocess.DEVNULL)
            fresh = open(path, "rb").read()
        finally:
            open(path, "wb").write(committed)
        same = strip(json.loads(committed)) == strip(json.loads(fresh))
        print(("PASS " if same else "FAIL ") + out)
        if not same:
            failed.append(out)
    if failed:
        sys.exit("results changed when regenerated: " + ", ".join(failed))


if __name__ == "__main__":
    main()
