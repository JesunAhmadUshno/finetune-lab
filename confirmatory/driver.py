"""Driver for the confirmatory run (lock docs/decisions/0006).

Runs units in the run-plan order (H1, H3/H4, H2), skipping units
whose output file already exists, so an interruption loses at
most the unit in flight. Logs each unit's wall time into
runs/gpu_ledger.json (the GPU-hour figure against the 96-hour
cap) and a status summary into runs/status.json.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
PY = os.path.join(os.path.dirname(HERE), ".venv", "bin", "python")
SMOKE = os.environ.get("CONF_SMOKE") == "1"
SEEDS = [0] if SMOKE else list(range(10))
BETAS = ["0", "0.04", "0.1"]


def ledger_path():
    return os.path.join(RUNS, "gpu_ledger.json")


def load_ledger():
    if os.path.exists(ledger_path()):
        with open(ledger_path()) as f:
            return json.load(f)
    return {"units": {}, "total_hours": 0.0}


def save_ledger(led):
    os.makedirs(RUNS, exist_ok=True)
    with open(ledger_path(), "w") as f:
        json.dump(led, f, indent=2)


def avail_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024 / 1024
    return 999.0


def wait_for_memory(min_gib=30.0, tries=10):
    for _ in range(tries):
        if avail_gib() >= min_gib:
            return True
        print(f"waiting for memory: {avail_gib():.0f} GiB available",
              flush=True)
        time.sleep(60)
    return avail_gib() >= min_gib


def run_unit(name, argv, done_file):
    if os.path.exists(done_file):
        print(f"SKIP {name}", flush=True)
        return True
    led = load_ledger()
    if led["total_hours"] >= 96.0:
        print(f"CAP REACHED before {name}: {led['total_hours']:.2f} h",
              flush=True)
        return False
    logdir = os.path.join(RUNS, "logs")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, name.replace("/", "_") + ".log")
    print(f"RUN {name} (ledger {led['total_hours']:.2f} h)", flush=True)
    env = dict(os.environ)
    env["HF_HOME"] = os.path.join(os.path.dirname(HERE), "hf_cache")
    if SMOKE:
        env["CONF_SMOKE"] = "1"
    t0 = time.time()
    rc = 1
    for attempt in range(2):
        wait_for_memory()
        with open(logpath, "a" if attempt else "w") as log:
            rc = subprocess.call([PY] + argv, cwd=HERE, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        if rc == 0 and os.path.exists(done_file):
            break
        print(f"retry {name} after rc={rc}", flush=True)
        time.sleep(60)
    hours = (time.time() - t0) / 3600.0
    led = load_ledger()
    led["units"][name] = {"hours": hours, "rc": rc,
                          "done_file_present": os.path.exists(done_file)}
    led["total_hours"] = sum(u["hours"] for u in led["units"].values())
    save_ledger(led)
    ok = rc == 0 and os.path.exists(done_file)
    print(f"{'OK' if ok else 'FAIL'} {name} {hours:.3f} h rc={rc}",
          flush=True)
    return ok


def main():
    os.makedirs(RUNS, exist_ok=True)
    R = RUNS
    run_unit("prep", ["prep_data.py"],
             os.path.join(HERE, "data", "manifest.json"))
    run_unit("eval/base", ["gsm8k_eval.py", "--tag", "base"],
             os.path.join(R, "evals", "base.summary.json"))
    # H1
    for s in SEEDS:
        for arm in ("full", "lora"):
            sub = "ckpt" if arm == "full" else "ckpt_adapter"
            run_unit(f"h1/{arm}/seed{s}",
                     ["sft_train.py", "--arm", arm, "--seed", str(s)],
                     os.path.join(R, "h1", arm, f"seed{s}", "metrics.json"))
            ck = os.path.join(R, "h1", arm, f"seed{s}", sub)
            run_unit(f"eval/h1_{arm}_seed{s}",
                     ["gsm8k_eval.py", "--model", ck,
                      "--tag", f"h1_{arm}_seed{s}"],
                     os.path.join(R, "evals", f"h1_{arm}_seed{s}.summary.json"))
    # pairs + H3/H4
    for s in SEEDS:
        run_unit(f"pairs/seed{s}", ["build_pairs.py", "--seed", str(s)],
                 os.path.join(R, "pairs", f"seed{s}", "counts.json"))
        sft = os.path.join(R, "h1", "full", f"seed{s}", "ckpt")
        run_unit(f"eval/pref_sft_seed{s}",
                 ["pref_eval.py", "--seed", str(s), "--model", sft,
                  "--tag", f"pref_sft_seed{s}"],
                 os.path.join(R, "evals", f"pref_sft_seed{s}.summary.json"))
        for arm in ("aligned", "misaligned"):
            run_unit(f"dpo/{arm}/seed{s}",
                     ["dpo_train.py", "--seed", str(s), "--arm", arm],
                     os.path.join(R, "dpo", arm, f"seed{s}", "metrics.json"))
        run_unit(f"eval/pref_dpo_seed{s}",
                 ["pref_eval.py", "--seed", str(s),
                  "--model", os.path.join(R, "dpo", "aligned",
                                          f"seed{s}", "ckpt"),
                  "--tag", f"pref_dpo_seed{s}"],
                 os.path.join(R, "evals", f"pref_dpo_seed{s}.summary.json"))
        run_unit(f"eval/h4_mis_seed{s}",
                 ["gsm8k_eval.py",
                  "--model", os.path.join(R, "dpo", "misaligned",
                                          f"seed{s}", "ckpt"),
                  "--tag", f"h4_mis_seed{s}"],
                 os.path.join(R, "evals", f"h4_mis_seed{s}.summary.json"))
    # H2
    for s in SEEDS:
        for b in BETAS:
            run_unit(f"grpo/beta{b}/seed{s}",
                     ["grpo_train.py", "--seed", str(s), "--beta", b],
                     os.path.join(R, "grpo", f"beta{float(b)}",
                                  f"seed{s}", "metrics.json"))
    led = load_ledger()
    status = {"total_hours": led["total_hours"],
              "units": {k: v for k, v in led["units"].items()},
              "smoke": SMOKE}
    with open(os.path.join(R, "status.json"), "w") as f:
        json.dump(status, f, indent=2)
    print("DRIVER DONE", led["total_hours"], flush=True)


if __name__ == "__main__":
    main()
