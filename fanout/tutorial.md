# Tutorial: P10 finetune-lab, course module

Module: P10-finetune-lab
By Jesun Ahmad Ushno
Date: 2026-11-05
License for this tutorial: CC BY 4.0. Code in the repo: Apache-2.0.

## What you will do

You will clone the finetune-lab repo, run all three levels, run
the test suite for each level, and read your own results against
the recorded v1.0 numbers. Every command below exists in the repo
as shipped at v1.0 and was run by the author of this module on
2026-11-05 to verify it. The project needs Python 3.11 or 3.12
with NumPy. No model download, no API, no credential, and no GPU
are needed.

All recorded numbers in this module come from synthetic NumPy
tasks. This is toy teaching work, not a real LLM: there is no
text, no tokenizer, and no open model run at any level. The
policy in Build and Deep is a tabular softmax over 8 prompts
and 4 actions, and Deep preferences are synthetic, from a
deterministic checker rule.

## Step 0: Clone and enter the repo

```text
git clone https://github.com/JesunAhmadUshno/finetune-lab.git
cd finetune-lab
```

## Step 1: Level 1, Learn, LoRA from scratch and GRPO advantages

Read level-1-learn/README.md first. It defines the synthetic
task exactly (512 train, 128 test, 4 classes, input dim 16, the
true change rank 2 by construction), the LoRA adapter (rank 2,
alpha 4.0, scaling 2.0, 40 trainable parameters against 64 for
full fine-tuning), the written RLHF pipeline, and the GRPO
against PPO note.

Run the Learn task set:

```text
python3 level-1-learn/run_learn.py
```

Run the Learn tests:

```text
python3 -m unittest tests.test_learn_finetune -v
```

Run the scaffold smoke test:

```text
python3 tests/test_smoke.py
```

Read your results:

- Expected runner lines: LoRA test accuracy before 0.5469 after
  0.9219; LoRA trainable parameters 40 of full 64; GRPO example
  groups 3.
- Expected summary: test accuracy 0.5469 (70 of 128) to 0.9219
  (118 of 128), test loss 1.0707 to 0.2534, frozen base weights
  unchanged, GRPO advantages hand checked for 3 groups of 4.
- Expected tests: 14 tests, all passing.
- Expected smoke test line: scaffold smoke test: PASS, 8 files
  present.

## Step 2: Level 2, Build, SFT pipeline and GRPO training

Read level-2-build/README.md and level-2-build/BUILD-SPEC.md.
The pipeline does data prep, minibatch training (batch 64, 60
epochs), checkpoint save and load with hash verification, and
an eval harness. The GRPO track trains the tabular policy
(group size 8, clip eps 0.2, KL beta 0.04, 60 iterations).
Read the split note in that README before comparing numbers:
the Build validation slice is the Learn test slice, and the
Build test slice is fresh held-out data.

Run the Build task set:

```text
python3 level-2-build/run_build.py
```

Run the Build tests:

```text
python3 -m unittest tests.test_build_finetune -v
```

Read your results:

- Expected runner lines: SFT test accuracy before 0.6797 after
  0.9141; SFT test loss before 1.0007 after 0.2032; GRPO eval
  reward before 0.207 after 0.9766; GRPO greedy accuracy before
  0.125 after 1.0.
- Expected summary: held-out test accuracy 0.6797 (87 of 128)
  to 0.9141 (117 of 128), checkpoint round-trip reproducing the
  after numbers exactly, GRPO fresh-sample mean reward 0.2070
  to 0.9766, greedy accuracy 0.125 (1 of 8) to 1.0000 (8 of 8),
  final mean KL to the reference policy 1.1795.
- Expected tests: 14 Build tests, all passing, 28 in total at
  that level.

## Step 3: Level 3, Deep, DPO, forgetting, hacking probe, QLoRA accounting

Read level-3-deep/README.md. The DPO run uses beta 0.1, a
frozen reference, and 300 full-batch steps on synthetic pairs
from the checker rule. QLoRA is accounting from parameter
counts, not measured GPU memory.

Run the Deep experiment:

```text
python3 level-3-deep/run_deep.py
```

Run the Deep tests:

```text
python3 -m unittest tests.test_deep_dpo -v
```

Read your results:

- Expected runner lines: DPO train margin before -0.1123 after
  1.6779; DPO held-out margin before -0.0654 after 0.8297; DPO
  held-out preference accuracy before 0.4375 after 1.0; task
  greedy accuracy before 0.125 after 1.0.
- Expected DPO loss: 0.6931 to 0.6076.
- Expected forgetting check: greedy checker accuracy 0.125 to
  1.0000, fresh-sample mean reward 0.2070 to 0.5430, no
  forgetting observed on that aligned task, named a weak test
  by construction in the README.
- Expected hacking probe: constant action-0 policy 0.2500;
  rejected-only suppression policy train preference accuracy
  1.0000 with held-out preference accuracy 0.4375; trained DPO
  policy not flagged.
- Expected QLoRA accounting, 7B row: 3,516,777,216 bytes
  against 14,000,000,000 full fp16 bytes, fraction 0.2512.
- Expected tests: 11 Deep tests, all passing, 39 in total.

## Step 4: Run the full test suite

```text
python3 -m unittest discover -s tests -v
```

Expected: 39 tests in total (14 Learn, 14 Build, 11 Deep), all
passing, plus the scaffold smoke test.

## Step 5: Check your understanding

1. Why does this project call itself toy work, not a real LLM
   fine-tune? (Answer: every level uses synthetic NumPy tasks.
   Learn and Build train one linear layer with a LoRA adapter,
   and Build and Deep train a tabular softmax policy over 8
   prompts and 4 actions. There is no text, tokenizer, sequence
   modelling, or open model run.)
2. Why can a rejected-only suppression policy score train
   preference accuracy 1.0000 without learning the task?
   (Answer: DPO train pairs reward raising the chosen action
   over one specific rejected action. Lowering only that
   rejected action raises every train margin, but the held-out
   rejected contrasts and the checker task do not move, which
   is why the probe reports held-out accuracy 0.4375 and greedy
   accuracy 0.2500 for that policy, and why this project always
   reports held-out pairs plus a task score.)
3. Why is the QLoRA 7B number 0.2512 not a memory measurement?
   (Answer: it is accounting: 3,500,000,000 bytes for a 4-bit
   base plus 16,777,216 adapter bytes, divided by
   14,000,000,000 fp16 bytes. Optimizer state, gradients,
   activations, and quantization scales are excluded, and no
   quantized model was loaded or measured.)

## Limits to keep with the numbers

- Synthetic data only, at every level. No real LLM is run.
- Preferences are synthetic, from a deterministic checker rule.
- QLoRA numbers are accounting, not measured GPU memory.
- The forgetting check is weak by construction: preferences
  are aligned with the retained task.
- Held-out in Deep means unseen rejected contrasts on known
  prompts only.
- All runs are on CPU at teaching scale, not on a GPU node.
