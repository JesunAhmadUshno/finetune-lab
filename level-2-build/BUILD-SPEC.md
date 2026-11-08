# BUILD-SPEC: finetune-lab Level 2, Build (P10)

Status: DONE 2026-11-05 (early, Sprint 3 pull). This spec states
what Build had to do, in the PROPOSAL.md (original scope) Build scope, and how each item
is evidenced. No item closes on a claim: code, tests, and measured
numbers only.

## Scope from PROPOSAL.md (original scope)

Build: one small GRPO run on an open model with a verifiable task
(math or code with a checker), scored before and after. Plus the
v1 core pipeline the project section names: a full fine-tune
pipeline (data build, train, evaluate before and after).

## How this Build meets it, and where it honestly does not

1. Full pipeline, DONE in pipeline.py: data prep (deterministic
   512 train, 128 validation, 128 held-out test split with a data
   card), SFT-style LoRA training (minibatch, batch 64, 60 epochs,
   frozen base, adapter only), checkpoint save and load with hash
   verification, and an eval harness (loss, accuracy, per-class
   accuracy, confusion matrix) reporting before and after on the
   held-out test set.
2. GRPO run with a verifiable task, DONE in grpo_build.py in toy
   form: a tabular categorical policy over 8 prompts, a checker
   reward (1.0 for the correct action, else 0.0), group advantages,
   a clipped surrogate policy loss, an exact KL term to the frozen
   reference policy, and measured reward improvement over 60
   iterations, scored before and after by the eval function in the
   same module.
3. Named gap, NOT hidden: the SPEC words say "on an open model"
   and the wider batch pattern scores with the P06 harness. This
   Build runs on a NumPy tabular toy policy, not an open language
   model, and scores with its own harness, not the P06 harness.
   That gap is a named limit (see the README limits), not a
   footnote. A real open-model GRPO run needs a GPU window and a
   model download; it is listed in the level README as the top
   follow-up and belongs with the Deep budget study.

## Acceptance checks

- run_build.py runs end to end and writes results_build.json.
  Running it twice gives a byte identical file.
- Unit tests in tests/test_build_finetune.py cover checkpoint
  round-trip (including rejection on a wrong base hash), eval
  harness arithmetic, batch coverage, GRPO clipping by hand, GRPO
  KL by hand, the checker, reward improvement, and determinism.
- The frozen base hash is identical before and after SFT training.
- The checkpoint round-trip reproduces the after-training test
  numbers exactly.
- No em dash in any file touched at this level.
