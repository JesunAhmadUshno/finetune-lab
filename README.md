# finetune-lab

**Question.** Do the LoRA, GRPO and DPO update rules behave as their
derivations predict, against baselines and across seeds, on synthetic
tasks small enough to check by hand?

**Headline.** On a synthetic NumPy classification task, a rank 2 LoRA
adapter (40 trainable weights) reaches mean held-out test accuracy
0.9039 (bootstrap 95% CI 0.8742 to 0.9305, 10 seeds), below full
fine-tuning of all 64 weights at 0.9391 (CI 0.9219 to 0.9555). No real
language model was trained.

**Field.** ML and AI research. **Job family.** ML research and
post-training engineering.

**Verdict.** Functional; not pre-registered. Tests pass in CI on Python
3.11 and 3.12, every results file is regenerated from its runner by one
command, and all four trainers have run end to end on CPU with their
outputs committed. No hypothesis was written before a run, so the repo
is capped at Functional until the confirmatory re-run drafted in
PROPOSAL.md is locked and run on a real open model.

Live dashboard: https://jesunahmadushno.github.io/finetune-lab/

## Reproduce

```text
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 scripts/check_results.py
```

check_results.py reruns all four runners (level-1-learn, level-2-build,
level-3-deep, rigor; a few seconds on one CPU core) and fails if any
recorded number changes. It ignores only the environment string and the
two Build adapter checkpoint hashes, which change with platform and
NumPy build (REPORT.md, section 6). To rewrite the files instead, run
each runner directly, for example `python3 rigor/run_rigor.py`.

## Lifecycle

- Question: DONE 2026-11-04
- Proposal: OPEN (not pre-registered; confirmatory re-run drafted, not locked)
- Baseline: DONE 2026-11-05
- Build: DONE 2026-11-05
- Evaluate: DONE 2026-11-07 (toy scale only, post hoc)
- Report: DONE 2026-11-08
- Review: OPEN
- Release: OPEN

## Layout

| Path | What it holds |
| --- | --- |
| PROPOSAL.md | Original scope (from SPEC.md), prior work (from NOVELTY.md), the draft confirmatory re-run; not locked |
| REPORT.md | Background, method, results, post hoc hypotheses scored, misses, threats, open work |
| level-1-learn/ | Baseline code: LoRA from scratch, GRPO advantages; README is the method note; results_learn.json is its evidence |
| level-2-build/ | Build code: SFT LoRA pipeline and GRPO track; results_build.json |
| level-3-deep/ | Evaluation code: DPO, retention check, reward hacking probe, QLoRA accounting; results_deep.json |
| rigor/ | 10-seed intervals, baselines, sweeps, ablations; results_rigor.json |
| scripts/check_results.py | Reruns every runner and checks the committed results are unchanged |
| tests/ | Unit tests and the layout test, run in CI |
| docs/decisions/ | Records of hard-to-reverse choices |
| site/ | The live dashboard, built from the results files and these docs |
| fanout/, PUBLICITY.md | Dated outreach drafts (2026-11-05) and the channel log |
| CHANGELOG.md, CITATION.cff, SECURITY.md, LICENSE | Project records |

Code stays in the stage folders (level-1-learn, level-2-build,
level-3-deep, rigor) because the tests, the runners and the dashboard
import from those paths (decision 0002). Each results JSON sits next to
the runner that writes it; there is no separate experiments/ or
results/ folder because the evidence is generated, not captured.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
