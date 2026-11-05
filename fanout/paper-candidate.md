# Paper candidate brief: the rejected-only suppression pattern in miniature

Project: P10 finetune-lab
Status: CANDIDATE, brief written 2026-11-05
Submission note: submission is Jesun's tap. Nothing is submitted by
this brief, and no venue has been contacted. The publications
tracker is updated by the coordinator, not by this file.

## Working title

Learning the contrast instead of the task: a rejected-only
suppression policy games DPO train preference accuracy to 1.0000
while held-out accuracy stays 0.4375, measured on a tabular
teaching policy

## Abstract draft

Based only on the measured v1.0 results recorded in the repo.
Numbers below are from synthetic data and are stated as such.
No real LLM was run, the policy is a toy NumPy tabular softmax,
and preferences are synthetic from a deterministic checker
rule.

Preference training is usually reported on the pairs it trained
on. This work asks what that reporting misses, using a small
system where every mechanism can be checked by hand. At Learn
level, LoRA from scratch on a synthetic task (512 train, 128
test, 4 classes) raised test accuracy from 0.5469 (70 of 128) to
0.9219 (118 of 128) with a rank 2 adapter of 40 trainable
parameters, and GRPO advantages were computed and hand checked
for 3 groups of 4. At Build level, an SFT-style pipeline raised
held-out test accuracy from 0.6797 (87 of 128) to 0.9141 (117 of
128), and a GRPO track on 8 checker prompts raised fresh-sample
mean reward from 0.2070 to 0.9766 with greedy accuracy from
0.125 (1 of 8) to 1.0000 (8 of 8) and a final mean KL of 1.1795.
At Deep level, a DPO run (beta 0.1, frozen reference, 300 steps)
on the Build tabular policy raised held-out mean margin from
-0.0654 to 0.8297 (+0.8951) and held-out preference accuracy
from 0.4375 (7 of 16) to 1.0000 (16 of 16), with DPO loss from
0.6931 to 0.6076. The central result is the probe: a policy
constructed by subtracting 5.0 from only the train rejected
action of each prompt, never raising the chosen action, reaches
train preference accuracy 1.0000 while held-out preference
accuracy stays 0.4375, greedy checker accuracy is 0.2500, and
mean reward is 0.3477. A constant action-0 policy scores 0.2500
on all surfaces and cannot game the signal, and the trained DPO
policy is not flagged by the same checks. The stack ships with
39 unit tests. The paper claims this miniature pattern and its
measurement protocol (train pairs, held-out rejected contrasts,
and an independent task score reported together), not behaviour
on real language models or human preference data.

## Method outline

1. Teaching policy: a tabular softmax over 8 prompts and 4
   actions, imported unmodified from the Build GRPO track, with
   one checker-correct action per prompt.
2. Pair construction, deterministic: the chosen response is the
   checker-correct action. Train rejected actions are at offset
   plus 1 modulo 4 (8 pairs). Held-out rejected actions are at
   offsets plus 2 and plus 3 modulo 4 (16 pairs), never seen as
   rejected in training.
3. DPO training: negative log sigmoid of beta times the
   reference adjusted margin, beta 0.1, frozen reference, full
   batch gradient descent, learning rate 0.5, 300 steps, with a
   hand-derived gradient checked by a hand case and by finite
   differences in the test suite.
4. Suppression construction: take the pre-DPO logits and
   subtract 5.0 from only the train rejected action of each
   prompt. The chosen action is never raised. Score this policy
   on train pairs, held-out pairs, greedy checker accuracy, and
   sampled mean reward.
5. Controls: a constant action-0 policy scored on the same four
   surfaces, and the trained DPO policy scored on the same four
   surfaces with a pre-stated flag rule (held-out accuracy more
   than 0.2 below train, or greedy accuracy below its pre-DPO
   value).
6. Forgetting surfaces: the same checker task scored before and
   after DPO (greedy accuracy, sampled mean reward, mean KL
   0.1833, untouched-action retention 1.0000), with the weakness
   of the aligned test stated as a limit, not a finding.
7. QLoRA context, accounting only: the stated 7B row totals
   3,516,777,216 bytes against 14,000,000,000 full fp16 bytes
   (fraction 0.2512), computed from parameter counts with
   optimizer state and other costs excluded and named.

## Results table

Measured 2026-11-05, synthetic data only, from the level READMEs
in the repo.

| Policy or level | Set | Headline result | Task score |
| --- | --- | --- | --- |
| Learn, LoRA | 128 test, synthetic | Accuracy 0.5469 to 0.9219, 40 of 64 parameters | Loss 1.0707 to 0.2534, 14 tests |
| Build, GRPO | 8 checker prompts | Mean reward 0.2070 to 0.9766 | Greedy 0.125 to 1.0000, KL 1.1795 |
| Deep, trained DPO | 16 held-out pairs | Margin -0.0654 to 0.8297, accuracy 0.4375 to 1.0000 | Greedy 0.125 to 1.0000, reward 0.2070 to 0.5430 |
| Probe, constant action-0 | Same pairs | Train and held-out preference accuracy 0.2500 | Greedy 0.2500, reward 0.2500 |
| Probe, rejected-only suppression | Same pairs | Train preference accuracy 1.0000, held-out 0.4375 | Greedy 0.2500, reward 0.3477 |

## Venue class note

Per SPEC.md, the paper angle for P10 is the GRPO against PPO
small-model study. This brief names a sharper angle found in the
shipped Deep work, the rejected-only suppression pattern, and
keeps the GRPO material as supporting results. Venue class is a
workshop or conference in ML training, evaluation, or alignment,
or a journal of the AJSE class, per the publications pipeline
tracker rules. One venue at a time. No venue is named as
submitted, accepted, or contacted in this brief. Fees, if any,
arise only after acceptance and remain Jesun's call under the
free-only rule.

## Limits the paper must carry

- Toy tabular policy, synthetic checker-rule preferences, and
  8 prompts only. The miniature pattern shows a reporting risk
  exists in this construction; it does not measure its size on
  real models, real text, or human preferences.
- Held-out means unseen rejected contrasts on known prompts,
  not unseen prompts or tasks. A stronger test on unseen
  prompts is in the v1.1 backlog.
- The suppression policy is constructed, not learned by an
  optimizer run. The claim is that the train contrast alone
  cannot distinguish it from learning, which the four-surface
  scoring demonstrates in this system.
- The forgetting check is weak by construction (aligned
  preferences, independent tabular rows) and is reported as a
  limit.
- QLoRA figures are accounting, not measured GPU memory.
- One beta, one learning rate, and one seed in Deep. No sweep
  is claimed.

## Submission gate

Submission is Jesun's tap and nothing is submitted. This brief is
the candidate only.
