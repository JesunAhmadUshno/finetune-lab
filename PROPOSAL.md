# Proposal: finetune-lab (P10)

Status: NOT LOCKED. This repo is not pre-registered. SPEC.md was
committed on 2026-11-04 (commit 164712d) before any code, but it set
scope and acceptance criteria only: no hypothesis, pass line or
analysis plan existed before the first run. No lock is claimed after
results exist. The confirmatory re-run in section 4 is a draft for
the Owner to settle and lock (tag proposal-v1) before it runs.

## 1. Question

Original scope question (SPEC.md): fine-tune a small open model with
LoRA on a custom instruction set, run GRPO on an open model with a
verifiable task, and publish before and after numbers.

Question this repo actually answers (README.md): do the LoRA, GRPO
and DPO update rules behave as their derivations predict, against
baselines and across seeds, on synthetic tasks small enough to check
by hand?

## 2. Original scope (SPEC.md, verbatim)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 5.
Hiring evidence: 5 of 5 (post-training duties at Cohere, Mistral, OpenAI; v2 rank 5) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

### Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

finetune-lab: EXPAND

Subject: fine-tuning and post-training. The CEO next-15 list puts this project at step 7: fine-tune a model with before and after eval numbers published [confirmed, cross-match Part 5].
v1 core [confirmed, v1]: LoRA fine-tune of a small open model on a custom instruction set; a full pipeline (data build, train, evaluate before and after); Deep adds a QLoRA memory study, a small DPO run, and forgetting checks.
Added track, reasoning and RL (register items 400, 402):
- Learn: draw the RLHF pipeline end to end from memory (SFT, reward model, policy run) and check the drawing against two public sources; write the GRPO against PPO note (what the group baseline replaces, what it costs).
- Build: one small GRPO run on an open model with a verifiable task (math or code with a checker), scored by the P06 harness before and after.
- Deep: a budget study in the P03 style: accuracy against training cost for the GRPO run, plus one failure log entry class (reward hacking seen in the run, or a clean statement that none appeared, with the checks named).
Fan-out: Book chapter "Teaching The Model Manners". Paper angle: the GRPO against PPO small-model study. Course module: P10-finetune-lab.

### Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.

## 3. Prior work and novelty (NOVELTY.md, verbatim)

Status: DRAFT, PROPOSED. Not verified. This text does not move into any public post, paper, or profile until the Jev claim_check gate passes on the exact text.

### Proposed claim

Against fine-tuning recipes in framework docs (TRL class): this project aims to differ in the way its SPEC.md section text defines (a from-zero or self-hosted build, measured in public, on a named home fleet, with the failure modes written down). The exact difference is stated at the project's wave, after the prior art is re-read.

### Prior art table

| Prior art | What it does | What this project does differently | Measured difference |
| --- | --- | --- | --- |
| fine-tuning recipes in framework docs (TRL class) | See DOSSIER.md, Past | PROPOSED, see the claim above | NO MEASUREMENT YET |

### Honest size

If the novelty proves small, this file will say so and name the real value instead (a cleaner reproduction, a better benchmark, a first public dataset). No novelty is invented to fill this section.

Primary sources for each method are listed in REPORT.md, section 8.

## 4. Confirmatory re-run (DRAFT, not locked)

Required by the portfolio standard for this repo. The settings in
brackets are decisions for the Owner, not facts.

- Model: one open model of roughly 0.5B to 1.5B parameters [which
  model].
- Compute: one GPU with at least 24 GB [which machine, how many
  GPU-hours].
- Data: a public instruction set and a public math word-problem set
  with a programmatic checker [which sets].
- Candidate hypotheses, carried from the post hoc set in REPORT.md:
  - H1: a LoRA adapter reaches held-out accuracy within [x] points of
    full fine-tuning at matched budget.
  - H2: the GRPO KL term trades reward for closeness to the
    reference: higher beta gives lower final KL and lower reward.
  - H3: DPO on aligned preferences improves held-out preference
    accuracy over the reference by at least [x].
  - H4: DPO on misaligned preferences lowers accuracy on an unrelated
    retained task by at least [x].
- Analysis: [n] seeds per arm, bootstrap 95% intervals, every run
  reported, baselines (full fine-tune, PPO with a critic) under the
  same budget.
- Pass lines and the CONFIRMED / REFUTED / INCONCLUSIVE rule:
  [to be set before the lock].
