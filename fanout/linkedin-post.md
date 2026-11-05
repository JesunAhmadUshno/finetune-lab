# LinkedIn post draft: P10 finetune-lab v1.0

Status: DRAFT NOT POSTED
Date drafted: 2026-11-05
Author: Jesun Ahmad Ushno
Rule for this platform: the link goes in the first comment, not in the
post body. The first comment line is at the end of this file.

## Post body

A fine-tuning result you cannot reproduce, probe, or explain is
a recipe, not evidence. So I built the post-training stack from
scratch, scored it before and after, and labelled the toy scale
in the post itself.

finetune-lab v1.0 is done: Learn, Build, and a compressed Deep,
in NumPy. Honest scope first: this is toy teaching work, not a
real LLM. No real text data, no open model run, no GPU run, and
Deep preferences are synthetic from a deterministic checker
rule.

Learn: LoRA from scratch (rank 2, 40 trainable parameters
against 64 for full fine-tuning) on a synthetic task, 512 train
and 128 test. Test accuracy 0.5469 (70 of 128) to 0.9219 (118
of 128), loss 1.0707 to 0.2534, base weights unchanged, plus
hand-checked GRPO advantages for 3 groups of 4. 14 tests. Grade
8,200 of 10,000.

Build: an SFT-style pipeline with data prep, minibatch training,
a hash-verified checkpoint, and an eval harness, plus a GRPO
track that trains a tabular policy. SFT held-out accuracy
0.6797 (87 of 128) to 0.9141 (117 of 128), loss 1.0007 to
0.2032, checkpoint round-trip exact. GRPO fresh-sample mean
reward 0.2070 to 0.9766, greedy accuracy 0.125 (1 of 8) to
1.0000 (8 of 8), final KL to the reference 1.1795 at beta 0.04,
named as drift, not hidden. 28 tests in total at that level.
Grade 8,300 of 10,000.

Deep (compressed): a DPO run on the Build tabular policy (beta
0.1, frozen reference, 300 steps). Held-out margin -0.0654 to
0.8297 (+0.8951), held-out preference accuracy 0.4375 (7 of 16)
to 1.0000 (16 of 16), train margin -0.1123 to 1.6779, loss
0.6931 to 0.6076. Forgetting check: greedy checker accuracy
0.125 to 1.0000 and mean reward 0.2070 to 0.5430, no forgetting
observed, and named a weak test by construction because the
preferences are aligned with the task. Reward hacking probe:
a constant action-0 policy scores 0.2500 and cannot game; a
rejected-only suppression policy games train preference
accuracy to 1.0000 while held-out stays 0.4375, the hacking
pattern in miniature; the trained DPO policy is not flagged.
QLoRA accounting (not measured GPU memory): the 7B row is
3,516,777,216 bytes against 14,000,000,000 full fp16 bytes,
fraction 0.2512. 39 tests in total. Grade 8,000 of 10,000, with
deductions named in the repo.

The honest limits, in the post and not in a footnote: toy
NumPy policy, not a real LLM; synthetic checker-rule
preferences; held-out means unseen rejected contrasts on known
prompts only; QLoRA is accounting only; and the rest of the
Deep scope is named as the v1.1 backlog in the repo, not
claimed as done.

If you run post-training: which check catches preference
training that learns the contrast instead of the task in your
stack, held-out pairs, a task score, or both?

## First comment (post this as the first comment, per the standing rule)

Repo, releases v0.1, v0.5, and v1.0, and the full tutorial: https://github.com/JesunAhmadUshno/finetune-lab
