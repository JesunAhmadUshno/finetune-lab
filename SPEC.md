# SPEC: finetune-lab (P10)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 5.
Hiring evidence: 5 of 5 (post-training duties at Cohere, Mistral, OpenAI; v2 rank 5) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

## Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

finetune-lab: EXPAND

Subject: fine-tuning and post-training. The CEO next-15 list puts this project at step 7: fine-tune a model with before and after eval numbers published [confirmed, cross-match Part 5].
v1 core [confirmed, v1]: LoRA fine-tune of a small open model on a custom instruction set; a full pipeline (data build, train, evaluate before and after); Deep adds a QLoRA memory study, a small DPO run, and forgetting checks.
Added track, reasoning and RL (register items 400, 402):
- Learn: draw the RLHF pipeline end to end from memory (SFT, reward model, policy run) and check the drawing against two public sources; write the GRPO against PPO note (what the group baseline replaces, what it costs).
- Build: one small GRPO run on an open model with a verifiable task (math or code with a checker), scored by the P06 harness before and after.
- Deep: a budget study in the P03 style: accuracy against training cost for the GRPO run, plus one failure log entry class (reward hacking seen in the run, or a clean statement that none appeared, with the checks named).
Fan-out: Book chapter "Teaching The Model Manners". Paper angle: the GRPO against PPO small-model study. Course module: P10-finetune-lab.

## Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.
