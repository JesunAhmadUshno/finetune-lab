# Release notes draft: finetune-lab

Status: DRAFT notes for releases that already exist. Dated 2026-11-05.
These notes summarise the existing GitHub releases. No new release is
created by this file.

Releases verified with gh release list on 2026-11-05:
v0.1, v0.5, and v1.0, at the URLs below.

## v0.1, Learn: LoRA from scratch and GRPO advantages

URL: https://github.com/JesunAhmadUshno/finetune-lab/releases/tag/v0.1

- LoRA from scratch in NumPy on a frozen base layer (rank 2,
  alpha 4.0, scaling 2.0), a hand-written full batch training
  loop, the written RLHF pipeline (SFT, reward model, policy
  run), and a GRPO against PPO note with hand-checked advantage
  examples. Synthetic task only. No real LLM, no GPU run, and
  no SFT or RL policy run at this level.
- Measured run: synthetic task (512 train, 128 test, 4 classes,
  input dim 16), test accuracy 0.5469 (70 of 128) to 0.9219
  (118 of 128), test loss 1.0707 to 0.2534, 40 trainable
  parameters against 64 for full fine-tuning, base weights
  unchanged.
- Tests: 14 unit tests in tests/test_learn_finetune.py, plus
  the scaffold smoke test.
- Grade: 8,200 of 10,000, per the batch ledger.
- Limits carried in the release: synthetic task, true change
  rank 2 by construction, one linear layer, GRPO advantages
  only with no policy training.

## v0.5, Build: full SFT LoRA pipeline and GRPO policy training

URL: https://github.com/JesunAhmadUshno/finetune-lab/releases/tag/v0.5

- A full SFT-style LoRA pipeline (data prep with a data card,
  minibatch training, a hash-verified checkpoint whose
  round-trip reproduces the after numbers exactly, and an eval
  harness with per-class accuracy and a confusion matrix), plus
  a GRPO track that trains a tabular toy policy with a clipped
  surrogate and a KL term. All NumPy, all synthetic, all on CPU.
  No real LLM, no real text data, and no open model run.
- Measured run, SFT: held-out test accuracy 0.6797 (87 of 128)
  to 0.9141 (117 of 128), test loss 1.0007 to 0.2032, after 60
  minibatch epochs. Measured run, GRPO: fresh-sample mean
  reward 0.2070 to 0.9766, greedy accuracy 0.125 (1 of 8) to
  1.0000 (8 of 8), final mean KL to the reference policy
  1.1795 at beta 0.04.
- Tests: 14 Build unit tests, 28 in total with Learn.
- Grade: 8,300 of 10,000, per the batch ledger.
- Limits carried in the release: synthetic tasks only, the SPEC
  open-model wording is not met, scoring is this repo's own
  harness not the P06 harness, validation peaks before the
  final epoch with no early stopping, and the checkpoint format
  is npz, not safetensors.

## v1.0, Deep: DPO run, forgetting checks, hacking probe, QLoRA accounting

URL: https://github.com/JesunAhmadUshno/finetune-lab/releases/tag/v1.0

- Compressed Deep under the December Compression Protocol: a
  small DPO run on the Build tabular GRPO policy (beta 0.1,
  frozen reference, 300 full-batch steps, hand-derived
  gradient), forgetting checks on the Build checker task, a
  three-part reward hacking probe, and a QLoRA memory
  accounting study. Toy NumPy tabular policy only, not a real
  LLM. Preferences are synthetic, from a deterministic checker
  rule. QLoRA is accounting, not measured GPU memory.
- Measured run: held-out mean margin -0.0654 to 0.8297
  (+0.8951), held-out preference accuracy 0.4375 (7 of 16) to
  1.0000 (16 of 16), train margin -0.1123 to 1.6779, DPO loss
  0.6931 to 0.6076. Forgetting check: greedy checker accuracy
  0.125 to 1.0000 and fresh-sample mean reward 0.2070 to
  0.5430, no forgetting observed on that aligned task, named a
  weak test by construction. Hacking probe: constant action-0
  policy 0.2500 and cannot game; rejected-only suppression
  policy games train preference accuracy to 1.0000 while
  held-out stays 0.4375; trained DPO policy not flagged. QLoRA
  accounting, 7B row: 3,516,777,216 bytes against
  14,000,000,000 full fp16 bytes, fraction 0.2512.
- Tests: 11 Deep unit tests, 39 in total with Learn and Build.
- Grade: 8,000 of 10,000, deductions itemized in the Deep README
  (total deductions 2,000).
- The rest of the v2 Deep scope is the v1.1 backlog in
  ROADMAP.md. It is not part of v1.0 and is not claimed by these
  notes.
