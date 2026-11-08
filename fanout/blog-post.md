# Teaching the Model Manners: Building finetune-lab in the Open

By Jesun Ahmad Ushno
Date: 2026-11-05
Status: DRAFT, written fan-out for P10 finetune-lab v1.0. Not posted.

## The problem

Fine-tuning is usually shown as a recipe: load a model, run a
framework, quote an accuracy. That hides the parts that fail in
practice: what the adapter can and cannot express, how far a
policy drifts from its reference, whether preference training
teaches the task or only the trained contrast, and what a
quantized run would actually cost in memory. This project builds
those pieces from scratch, small enough to read and check by
hand, and scores every level before and after.

This project, P10 in Roadmap v3, builds fine-tuning and
post-training in three levels: LoRA from scratch and GRPO
advantages; then a full SFT-style LoRA pipeline and a GRPO track
that trains a toy policy; then a compressed Deep with a DPO run,
forgetting checks, a reward hacking probe, and a QLoRA memory
accounting study. Labelled honestly at every level: this is a
toy NumPy tabular and linear teaching work, not a real LLM.
Synthetic preferences come from a deterministic checker rule.
No real LLM, no real text data, and no open model was run. All
work ran on Muse VM CPU, not on the BlackBox GPU node.

Repository: https://github.com/JesunAhmadUshno/finetune-lab

## Past, present, future (the dossier, condensed)

Past: fine-tuning recipes in framework docs (the TRL class)
already exist. The prior art and the problem history are
recorded in REPORT.md (background) and PROPOSAL.md (prior work) in the repo. The novelty
claim there stays marked PROPOSED. This post makes no novelty
claim beyond the measured work below. The methods are the
standard published ones, rebuilt from scratch in NumPy for
teaching: LoRA, GRPO, DPO, and QLoRA studied as accounting.

Present: v1.0 closes three levels on 2026-11-05. Learn built
LoRA from scratch and hand-checked GRPO advantages. Build added a full SFT-style pipeline with
checkpointing and an eval harness, plus a GRPO track that
trains a tabular toy policy. Compressed
Deep added a DPO run on the Build tabular policy, forgetting
checks, a reward hacking probe, and a QLoRA accounting study.

Future: the rest of the v2 Deep scope is named in REPORT.md (open work) in
the repo as the v1.1 backlog. It includes a real open-model
run, a measured QLoRA study, stronger misaligned forgetting
checks, a GRPO budget and KL beta study, DPO hyperparameter
sweeps, human or model preference data, and scoring with the
P06 harness. None of that is done, and none of it is claimed
here.

## Main results claim

At v1.0, on synthetic NumPy tasks only with no real LLM and no real text data: Learn passed 14 tests on a synthetic task (512 train, 128 test, 4 classes, input dim 16), where a rank 2 LoRA adapter with 40 trainable parameters against 64 for full fine-tuning raised test accuracy from 0.5469 (70 of 128) to 0.9219 (118 of 128) with test loss from 1.0707 to 0.2534 after 200 full batch epochs, the frozen base weights unchanged, and GRPO advantages were computed and hand checked for 3 stated reward groups of 4; Build passed 14 tests (28 tests in total at that level) with an SFT pipeline (512 train, 128 validation, 128 held-out test) that raised held-out test accuracy from 0.6797 (87 of 128) to 0.9141 (117 of 128) with test loss from 1.0007 to 0.2032 after 60 minibatch epochs and a checkpoint round-trip reproducing the after numbers exactly, and a GRPO track on 8 checker prompts (group size 8, clip eps 0.2, KL beta 0.04, 60 iterations) that raised fresh-sample mean reward from 0.2070 to 0.9766 with greedy accuracy from 0.125 (1 of 8) to 1.0000 (8 of 8) and a final mean KL to the reference policy of 1.1795; compressed Deep passed 11 tests (39 tests in total) with a small DPO run on the Build tabular GRPO policy (beta 0.1, frozen reference, 300 full-batch steps) that raised held-out mean margin from -0.0654 to 0.8297, a change of +0.8951, and held-out preference accuracy from 0.4375 (7 of 16) to 1.0000 (16 of 16), with train margin from -0.1123 to 1.6779 and DPO loss from 0.6931 to 0.6076, a forgetting check where greedy checker accuracy rose from 0.125 to 1.0000 and fresh-sample mean reward rose from 0.2070 to 0.5430 with no forgetting observed on that aligned task, a reward hacking probe where a constant action-0 policy scored 0.2500 and could not game the signal while a rejected-only suppression policy gamed train preference accuracy to 1.0000 while held-out preference accuracy stayed 0.4375 and the trained DPO policy was not flagged, and a QLoRA accounting study (labelled accounting, not measured GPU memory) where the stated 7B row totalled 3,516,777,216 bytes against 14,000,000,000 bytes for full fp16 weights, a fraction of 0.2512.

Every number in that claim comes from a run recorded in the
level READMEs and results files in the repo, on synthetic data.
Read the limits section below before quoting any of it.

## Results by level

### Level 1, Learn (release v0.1)

Learn builds LoRA from scratch in NumPy: a frozen base layer,
an adapter W0 plus scaling times (A @ B) with rank 2, alpha
4.0, scaling 2.0, A of shape (16, 2) and B of shape (2, 4), B
starting at zero so the adapted model starts exactly as the
base. Training is a hand-written full batch gradient descent
loop on mean cross entropy. The level also writes the RLHF
pipeline end to end (SFT, reward model, policy run) and the
GRPO against PPO note: GRPO replaces the learned critic with
the group mean reward as the baseline, at the cost of sampling
and scoring a whole group per prompt.

| Measure | Result |
| --- | --- |
| Synthetic task | 512 train, 128 test, 4 classes, input dim 16 |
| Test accuracy | 0.5469 (70 of 128) to 0.9219 (118 of 128) |
| Test loss | 1.0707 to 0.2534 |
| Trainable parameters | 40 of 64 full, fraction 0.625 |
| Base weights | Unchanged, hash before equals hash after |
| GRPO examples | 3 stated reward groups of 4, hand checked |
| Tests | 14 |
| Determinism | results_learn.json identical across two runs |

Limits are itemized in the Learn README.

### Level 2, Build (release v0.5)

Build turns the Learn idea into a pipeline: data prep with a
data card, minibatch SFT-style LoRA training (batch 64, 60
epochs), a checkpoint with hash verification whose round-trip
reproduces the after numbers exactly, and an eval harness with
per-class accuracy and a confusion matrix. The GRPO track
trains the tabular toy policy with a clipped surrogate (eps
0.2) and a KL term (beta 0.04). Note the split honestly: the
Build validation slice is the slice Learn used as its test
set, and the Build test slice is fresh held-out data, so Build
and Learn test numbers are not the same set.

| Measure | Result |
| --- | --- |
| SFT held-out test accuracy | 0.6797 (87 of 128) to 0.9141 (117 of 128) |
| SFT held-out test loss | 1.0007 to 0.2032 |
| GRPO fresh-sample mean reward | 0.2070 to 0.9766 |
| GRPO greedy accuracy | 0.125 (1 of 8) to 1.0000 (8 of 8) |
| GRPO train group reward | 0.2656 first iteration to 0.9219 last |
| Final mean KL to reference | 1.1795 nats at beta 0.04 |
| Tests | 14 Build, 28 in total at that level |

Limits are itemized in the Build README.
Two limits are named in that README, not hidden: validation
accuracy reaches 0.9375 at epoch 10 and ends at 0.9141, so the
run mildly overfits and ships the final epoch adapter with no
early stopping; and at beta 0.04 the KL term does not hold the
policy near the reference.

### Level 3, Deep, compressed (release v1.0)

Deep runs DPO on the Build tabular policy. Preference pairs
are synthetic, from a deterministic checker rule: the chosen
response is the checker-correct action, train rejected actions
are at offset plus 1 modulo 4 (8 pairs), and held-out rejected
actions are at offsets plus 2 and plus 3 (16 pairs), which
never appear as rejected in training. The loss is the negative
log sigmoid of beta times the reference adjusted margin, beta
0.1, with a hand-derived gradient checked against finite
differences.

| Measure | Before | After | Change |
| --- | --- | --- | --- |
| DPO loss (train pairs) | 0.6931 | 0.6076 | -0.0855 |
| Train mean margin (8 pairs) | -0.1123 | 1.6779 | +1.7902 |
| Held-out mean margin (16 pairs) | -0.0654 | 0.8297 | +0.8951 |
| Held-out preference accuracy | 0.4375 (7 of 16) | 1.0000 (16 of 16) | +0.5625 |
| Greedy checker accuracy | 0.125 (1 of 8) | 1.0000 (8 of 8) | +0.875 |
| Fresh-sample mean reward | 0.2070 | 0.5430 | +0.3360 |

Forgetting check: no forgetting was observed on the Build
GRPO task, and the level README names this a weak test by
construction. The preferences are aligned with the checker
task, so training and the retained task pull in the same
direction, and the tabular rows are independent. Mean KL from
the after policy to the pre-DPO policy is 0.1833 nats, and
untouched-action retention is 1.0000 (8 of 8).

Reward hacking probe: a constant action-0 policy scores
0.2500 on train, held-out, and greedy surfaces and cannot game
the signal. A rejected-only suppression policy, which lowers
only the train rejected action and never raises the chosen
action, games train preference accuracy to 1.0000 while
held-out preference accuracy stays 0.4375 and greedy accuracy
is only 0.2500: the hacking pattern found in miniature. The
trained DPO policy is not flagged by the probe checks.

QLoRA accounting: bytes are computed from stated parameter
counts, not measured GPU memory. For the stated 7B row (32
layers, hidden 4096, rank 16 adapter on query and value
projections), the 4-bit base is 3,500,000,000 bytes plus
16,777,216 adapter bytes, totalling 3,516,777,216 bytes
against 14,000,000,000 bytes for full fp16 weights, fraction
0.2512. Optimizer state, gradients, activations, and
quantization scales are excluded, and that exclusion is stated
in the code and the README.

Tests: 11 Deep tests, 39 in total with Learn and Build.
Determinism: results_deep.json identical across two runs.

Limits are itemized in the Deep README: a toy NumPy tabular
policy, not a real LLM; synthetic preferences from a
deterministic checker rule; QLoRA as accounting, not measured
GPU memory; runs on Muse VM CPU, not the BlackBox GPU node; a
forgetting test weak by construction; one beta, one learning
rate, and one seed with no sweep; and held-out meaning unseen
rejected contrasts on known prompts only.

## Honest limits

- Toy work, not a real LLM, at every level. No text, no
  tokenizer, no sequence modelling, no real instruction data,
  and no open model run. The policy in Build and Deep is a
  tabular softmax over 8 prompts and 4 actions.
- Preferences in Deep are synthetic, generated by a stated
  deterministic checker rule, not human or model preference
  data.
- The SFT task makes the true change from the base rank 2 by
  construction, so a rank 2 adapter can express it. Real tasks
  do not promise that.
- Held-out in Deep means unseen rejected contrasts on known
  prompts, not unseen prompts, unseen tasks, or text.
- The forgetting check is weak by construction, for the
  reasons named above. It does not prove DPO would preserve an
  unrelated capability on a real model, where capabilities
  share weights.
- QLoRA numbers are weight-storage accounting only. No
  quantized model was loaded or trained, and no GPU memory was
  measured.
- Greedy accuracy reaches 1.0000 in Deep while fresh-sample
  mean reward is only 0.5430. Reporting greedy alone would
  overstate the run; both are reported.
- One beta, one learning rate, and one seed configuration in
  Deep, and a fixed learning rate in Learn and Build. No sweep
  is claimed.
- Compressed Deep is not the full v2 Deep scope. The remainder
  is the named v1.1 backlog in REPORT.md (open work).
- The runs executed on the Muse VM, CPU, by this project's
  CPU teaching scale design, not on the BlackBox GPU node.
  That limit is named in each level README.

## Where to go next

- Tutorial: fanout/tutorial.md in the repo, a step-by-step
  module that runs all three levels and the full test suite.
- Releases: v0.1 (Learn), v0.5 (Build), v1.0 (Learn, Build,
  compressed Deep) on the GitHub repo linked above.
- Book chapter: "Teaching The Model Manners" per PROPOSAL.md (original scope), held
  as a draft angle in this fan-out set.
- License: code Apache-2.0, tutorials CC BY 4.0.
