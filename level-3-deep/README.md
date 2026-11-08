# Level 3, Deep: finetune-lab

Status: DONE (2026-11-05, Sprint 3 early pull under the December
compression protocol: the single hardest capability in the SPEC
Deep scope, built, tested, benchmarked, with failure modes
written; the rest of the v2 Deep scope is the v1.1 backlog in
the root REPORT.md (open work)). The Deep core here is a small DPO run
with forgetting checks and a QLoRA memory study, plus the
reward hacking probe the SPEC Deep text asks for as its failure
log entry. All NumPy, all synthetic, all on Muse VM CPU.

## Summary claim (beside its measured table)

Deep DONE: on a synthetic NumPy tabular policy only (8 prompts,
4 actions, reused from Build), a DPO run (beta 0.1, frozen
reference, 300 full-batch steps) raised the held-out preference
margin from -0.0654 to 0.8297 and held-out preference accuracy
from 0.4375 (7 of 16) to 1.0000 (16 of 16), with train margin
from -0.1123 to 1.6779, greedy checker accuracy from 0.125
(1 of 8) to 1.0000 (8 of 8) and no forgetting on that aligned
task (change +0.875 greedy, +0.336 mean reward), and a QLoRA
accounting study (not measured GPU memory) putting a 7B-scale
model's 4-bit base plus rank 16 adapter at 3,516,777,216 bytes
against 14,000,000,000 bytes for fp16 full weights; all on
Muse VM CPU with no real LLM.

| Measure | Before | After | Change |
| --- | --- | --- | --- |
| DPO loss (train pairs) | 0.6931 | 0.6076 | -0.0855 |
| Train pairs mean margin (8 pairs) | -0.1123 | 1.6779 | +1.7902 |
| Train pairs preference accuracy | 0.2500 (2 of 8) | 1.0000 (8 of 8) | +0.7500 |
| Held-out pairs mean margin (16 pairs) | -0.0654 | 0.8297 | +0.8951 |
| Held-out pairs preference accuracy | 0.4375 (7 of 16) | 1.0000 (16 of 16) | +0.5625 |
| Greedy checker accuracy (forgetting task) | 0.125 (1 of 8) | 1.0000 (8 of 8) | +0.875 |
| Fresh-sample mean reward (forgetting task) | 0.2070 | 0.5430 | +0.3360 |

Source of every number: level-3-deep/results_deep.json,
produced by run_deep.py. Running run_deep.py twice gives a
byte identical file (sha256 2e7c53a1a717e0a0d3512157468e
20fc611d140e6edbc00170b753ddb4b7859b).

## The DPO run (dpo_deep.py)

Build code reuse: the policy, checker, correct actions, and
evaluator are imported from level-2-build/grpo_build.py,
unmodified (attribution also in the module docstring). The
initial policy is the Build init_policy (seed 21, small seeded
tilt). The reference policy is a frozen copy of it.

- Preference pairs, deterministic rule: the chosen response is
  the checker-correct action. Train pairs use the wrong action
  at offset plus 1 modulo 4 (8 pairs, one per prompt). Held-out
  pairs use the wrong actions at offsets plus 2 and plus 3
  (16 pairs), which never appear as rejected in training.
- Loss: negative log sigmoid of beta times the reference
  adjusted margin, beta 0.1, averaged over train pairs.
- Gradient, hand-derived: per pair, the derivative of the
  reference adjusted margin with respect to that prompt's
  logits is exactly one-hot(chosen) minus one-hot(rejected),
  because the softmax Jacobian terms cancel between the two
  log-probabilities of the same row. Tests check the gradient
  against a hand-computed case and against finite differences.
- Training: full-batch gradient descent, learning rate 0.5,
  300 steps. Loss falls from 0.6931 to 0.6076 while the train
  margin grows steadily in the history table in the results
  file. The loss moves less than the margin because beta 0.1
  compresses the sigmoid argument; that is the stated beta
  doing its job, not a stalled run.

## Forgetting checks

The pre-DPO policy task is the Build GRPO checker task, scored
by the Build evaluator (eval seed 41, 32 samples per prompt)
before and after DPO:

- Greedy accuracy: 0.125 (1 of 8) before, 1.0000 (8 of 8)
  after, change +0.875.
- Fresh-sample mean reward: 0.2070 before, 0.5430 after,
  change +0.3360.
- Mean KL from the after policy to the pre-DPO policy: 0.1833
  nats. The policy moved, and the movement is measured.
- Untouched-action retention: the relative ordering of the
  two held-out rejected actions is preserved on 8 of 8
  prompts (retention 1.0000).

Honest naming: no forgetting was observed, and the change on
the task is positive, but this is a weak forgetting test by
construction, and the weakness is named, not hidden. The
preferences are aligned with the checker task (the chosen
action is the correct action), so DPO training and the task
pull in the same direction. A misaligned or multi-task
preference set would be a stronger forgetting test; it is in
the v1.1 backlog. The retention number is also partly
structural: tabular rows are independent and DPO updates only
the chosen and train rejected logits of each row, so the two
untouched logits keep their order by construction.

## QLoRA memory study (memory_study.py)

Accounting, not measurement. No GPU memory is measured. Bytes
are computed in code from stated parameter counts: full
fine-tune weights at 2 bytes per parameter (fp16), a rank 16
LoRA adapter adapting the query and value projections at
2 bytes per adapter parameter, and a QLoRA total of the base
at 0.5 bytes per parameter (4-bit) plus the same adapter.
Optimizer state, gradients, activations, fp32 master copies,
and quantization scales are excluded; the exclusions are in
the code and here.

| Model (stated size) | Full fp16 bytes | LoRA adapter bytes | LoRA total bytes | 4-bit base bytes | QLoRA total bytes | QLoRA fraction of full |
| --- | --- | --- | --- | --- | --- | --- |
| small-125M (12 layers, hidden 768) | 250,000,000 | 1,179,648 | 251,179,648 | 62,500,000 | 63,679,648 | 0.2547 |
| medium-1B (24 layers, hidden 2048) | 2,000,000,000 | 6,291,456 | 2,006,291,456 | 500,000,000 | 506,291,456 | 0.2531 |
| large-7B (32 layers, hidden 4096) | 14,000,000,000 | 16,777,216 | 14,016,777,216 | 3,500,000,000 | 3,516,777,216 | 0.2512 |
| xl-13B (40 layers, hidden 5120) | 26,000,000,000 | 26,214,400 | 26,026,214,400 | 6,500,000,000 | 6,526,214,400 | 0.2510 |

Reading: at weight-storage accounting, a 4-bit base plus the
adapter holds the weights in about a quarter of the fp16
full bytes, and the adapter alone is 0.10 to 0.47 percent of
the full bytes across these sizes. Because optimizer state is
excluded, the full column is a floor; a real training run's
gap between full fine-tuning and QLoRA is larger than this
table, in QLoRA's favour, and this table does not claim it.

## Reward hacking probe

Three named checks, run in code, outcomes recorded:

1. Constant policy probe: a policy with almost all mass on
   action 0 for every prompt scores train preference accuracy
   0.2500, held-out preference accuracy 0.2500, and greedy
   accuracy 0.2500. A fixed response cannot game this signal,
   because the chosen action varies by prompt.
2. Rejected-only suppression probe: subtracting 5.0 from only
   the train rejected action of each pre-DPO prompt (the
   chosen action is never raised) games the train contrast
   completely: train preference accuracy 1.0000. The gaming
   is exposed by the other surfaces: held-out preference
   accuracy stays 0.4375 and greedy accuracy is only 0.2500,
   with mean reward 0.3477. This is the reward hacking seen
   in this run, in miniature: a policy can maximise the
   trained preference contrast without learning the task,
   and only held-out contrasts plus a task score catch it.
3. Trained policy check: the actual DPO policy is not
   flagged. Held-out preference accuracy 1.0000 is not below
   train 1.0000 by the 0.2 margin, and greedy accuracy rose
   from 0.125 to 1.0000 instead of falling.

## Failure modes

- Aligned-preference forgetting weakness: covered above. The
  forgetting check can only show no forgetting on an aligned
  task; it does not prove DPO would preserve an unrelated
  capability, and on a real language model unrelated
  capabilities share weights, unlike these independent rows.
- Train-contrast gaming: the suppression probe shows the
  trained contrast alone is gameable. Any DPO result reported
  on train pairs only would be unsafe; this level reports
  held-out pairs and a task score for exactly that reason.
- Margin without probability mass: greedy accuracy reaches
  1.0000 while fresh-sample mean reward is only 0.5430. The
  margins are still small in probability terms (beta 0.1,
  300 steps), so sampled behaviour lags greedy behaviour.
  Reporting greedy alone would overstate the run.
- Loss compression at small beta: at beta 0.1 the loss moves
  only 0.6931 to 0.6076 while margins move by more than a nat.
  Reading the loss alone would understate the run. Both are
  reported.
- Accounting versus reality: the memory study excludes
  optimizer state and quantization scales, and no 4-bit model
  was run. Treating the table as measured GPU memory would
  be wrong; it is a weight-storage comparison only.
- Ceiling by construction: 8 prompts and 4 actions saturate
  fast. Held-out here means unseen rejected contrasts on
  known prompts, not unseen prompts, unseen tasks, or text.

## Honest limits

- Toy NumPy tabular policy, not a real LLM. No text, no
  tokenizer, no sequence modelling, no human or model
  preference data: preferences are synthetic, generated by
  the stated deterministic checker rule.
- QLoRA is studied as accounting only. No quantized model is
  loaded or trained in this level.
- Run on the Muse VM, CPU, by this project's CPU teaching
  scale design, not on the BlackBox GPU node. The CPU
  location is a named limit, not a silent
  substitution.
- One beta, one learning rate, one seed configuration. No
  sweep is claimed.

## Limits named at review

- Toy NumPy tabular policy, not a real LLM, no text or sequence modelling.
- Synthetic preferences from a deterministic checker rule, not human or model preference data.
- QLoRA memory study is accounting from parameter counts, not measured GPU memory, and no 4-bit model was run.
- Run on the Muse VM CPU, not the BlackBox GPU node, by this project's CPU teaching design.
- Forgetting test is weak by construction (aligned preferences, independent tabular rows).
- One beta, one learning rate, one seed configuration, no hyperparameter sweep.
- Held-out means unseen rejected contrasts on known prompts only, not unseen prompts or tasks.

## Run it

    python3 level-3-deep/run_deep.py
    python3 -m unittest discover -s tests

Tests: 39 total in the repo suite, 11 of them new in
tests/test_deep_dpo.py, covering the DPO loss hand case, the
gradient by finite differences, the pair rule, margin
improvement, forgetting change arithmetic, the suppression
probe, and the memory accounting arithmetic.
