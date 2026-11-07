# Research review: finetune-lab

Date: 2026-11-07. Scope: the whole repo at branch site/live-dashboard
(Learn, Build, Deep, and the new rigor/ study). Graded against a
doctoral research rubric. Each criterion gets a grade before this
change and after it, what was fixed here, and for every gap left
open, the exact compute or data it needs. Numbers are quoted from
the results files named beside them; nothing here is new evidence
beyond those files.

## Summary

finetune-lab is a careful teaching-scale study of post-training
methods (LoRA, SFT, GRPO, DPO, QLoRA accounting) on synthetic NumPy
tasks. Its strengths are determinism, hand-checked maths, unit
tests, and unusually plain naming of its own limits. Its central
weakness is unchanged by this review and cannot be fixed on a CPU:
no real language model is ever trained, so no claim transfers to
LLM fine-tuning. Within the toy scope, this change closes most of
the statistical and baseline gaps: every headline number now has a
10-seed interval, full fine-tuning and rank baselines exist, the KL
beta and DPO hyperparameters are swept, forgetting is tested with
misaligned preferences and with shared weights, and a defect in the
Build GRPO gradient was found and documented.

| Criterion | Before | After | Status |
| --- | --- | --- | --- |
| Research question | 5/10 | 6/10 | Partly open (needs a real model to answer the SPEC question) |
| Hypotheses | 3/10 | 6/10 | Stated post hoc below, labelled as such |
| Method | 7/10 | 8/10 | GRPO gradient defect found and documented |
| Baselines and ablations | 2/10 | 7/10 | Fixed at toy scale; PPO with a critic still open |
| Statistics and uncertainty | 1/10 | 7/10 | Fixed: 10 seeds, bootstrap intervals |
| Reproducibility | 7/10 | 8/10 | CI runs all tests; numpy pinned; platform note added |
| Threats to validity | 6/10 | 7/10 | Expanded below |
| Related work | 2/10 | 5/10 | Primary sources listed; no positioning against prior results |
| Limitations | 8/10 | 8/10 | Already strong; carried into the site |

## 1. Research question

SPEC.md asks for a LoRA fine-tune of a small open model on a custom
instruction set, a GRPO run on an open model with a verifiable math
or code task, and a GRPO against PPO small-model study. The repo
answers a narrower question: do the update rules of LoRA, GRPO and
DPO behave as their derivations say on tasks small enough to check
by hand? That narrower question is answered well. The SPEC question
is not answered at all.

Open, and what it needs: one open model of roughly 0.5B to 1.5B
parameters, a GPU with at least 24 GB of memory, a few GPU-hours per
method (an estimate, not measured), a public instruction set and a
public math word-problem set with a programmatic answer checker.
This is ROADMAP item 1.

## 2. Hypotheses

No hypothesis was written before any run. Stated now, post hoc, and
scored against the files:

- H1: a rank-2 adapter on a rank-2 target change reaches held-out
  accuracy close to full fine-tuning. Partly supported: 10-seed mean
  test accuracy 0.9039 for rank 2 against 0.9391 for full
  fine-tuning (rigor/results_rigor.json, sft_baselines). The adapter
  is clearly better than the base and clearly below full tuning.
- H2: GRPO's KL term trades reward for closeness to the reference.
  Supported: at 60 iterations, beta 0 gives mean reward 0.9637 and KL
  1.1980; beta 0.1 gives 0.8859 and 0.9544 (grpo_beta_budget).
- H3: DPO on aligned preferences generalises to unseen rejected
  contrasts. Supported: held-out preference accuracy 1.0000 on all
  10 seeds (seeds.dpo).
- H4: DPO on preferences that contradict the task removes the task
  where they disagree, and with shared weights the damage spreads.
  Supported: tabular greedy accuracy 0.0000 on misaligned prompts;
  shared-weight unseen-prompt greedy accuracy falls from 0.5375
  (aligned) to 0.2500 with 4 of 8 train prompts misaligned
  (misaligned_forgetting, shared_weight_dpo).

Because these are post hoc, they are descriptive. A preregistered
hypothesis set belongs with the real-model run in section 1.

## 3. Method

The derivations are stated in module docstrings and the DPO gradient
is checked by finite differences in tests/test_deep_dpo.py. The
QLoRA arithmetic is exact and checked.

Defect found in this review: level-2-build/grpo_build.py reports the
loss -mean(surrogate) + beta * mean KL, but its gradient divides
prompt p's surrogate term by the running sample count (p + 1) x 8
instead of the total 64, and adds the KL gradient without averaging
over prompts. Prompt 0 therefore steps 8 times harder than prompt 7,
and the KL pull is 8 times what the stated loss implies.
tests/test_rigor.py shows the Build gradient does not match finite
differences of its own loss, and rigor_study.grpo_grad_corrected
does. The recorded Build and Deep numbers are kept as run, because
rewriting them would hide the history; a corrected ablation reaches
mean reward 0.9180 (CI 0.9027 to 0.9332) against 0.9465 for the
recorded rule over 10 seeds (grpo_ablations). Fixed here: found,
tested, documented, quantified. The live GRPO panel on the site
mirrors the recorded rule and offers the corrected one as a toggle.

Also noted: the SFT pipeline ships the final-epoch adapter though
validation peaks earlier. Best-validation epoch selection now lifts
10-seed mean test accuracy from 0.9039 to 0.9219 (seeds.sft). Fixed
as an evaluated option in rigor/; the Build pipeline is unchanged.

## 4. Baselines and ablations

Before: none. Every result was before-against-after on one method.

Fixed here (rigor/README.md has the tables):

- Full fine-tuning of all 64 weights against LoRA ranks 1 to 4 on
  the same splits and schedule.
- GRPO without clipping, without the KL term, a plain REINFORCE
  update with the same group baseline, and the corrected gradient.
- A 36-setting DPO grid over beta, learning rate and steps.
- KL beta sweep for GRPO with reward and KL at six budgets.

Open, and what it needs: a real PPO baseline with a learned critic
on the same open model and task as section 1 (same GPU budget again,
roughly doubled for the critic), and a DPO against GRPO head-to-head
at matched compute on that task. The tabular REINFORCE variant is a
stand-in and is labelled as one.

## 5. Statistics and uncertainty

Before: one seed configuration everywhere, no intervals.

Fixed here: every headline number has a 10-seed mean, sample
standard deviation and a percentile bootstrap 95% interval of the
mean (10,000 resamples). Seed offset 0 reproduces the recorded
numbers exactly, so the intervals extend the recorded runs rather
than replace them. Readings that changed: the recorded GRPO reward
(0.9766) is the best of 10 seeds, the typical value is 0.9465 (CI
0.9340 to 0.9590); the recorded DPO held-out margin (0.8297) is the
lowest of 10 seeds, the mean is 0.8832.

Remaining caveat, not fixable by more CPU in a meaningful way: with
10 seeds a bootstrap interval runs narrow, and the evaluation sets
are tiny (8 prompts, 16 pairs, 128 test vectors). Intervals here
describe seed spread on toy tasks, not population uncertainty for a
real model. No significance tests are claimed.

## 6. Reproducibility

Strengths: fixed seeds, no timing fields, and each runner rewrites
its JSON identically on one platform. Every level now has a
one-command rerun, including python3 rigor/run_rigor.py (about 3
seconds).

Fixed here:

- CI ran only the scaffold smoke test. It now runs the full suite
  (55 tests, under a second) after installing pinned dependencies.
- requirements.txt pins numpy==1.26.4, the version the level results
  record.

Platform note found in this review: rerunning level-2-build/run_build.py
under NumPy 2.4.5 on an arm64 machine reproduces every rounded metric
in results_build.json but changes the two adapter checkpoint hashes
(floating point summation order differs by platform). The "byte
identical" claim holds per platform, not across platforms. The
rigor file records its own environment (Python 3.12.3, NumPy 2.4.5)
and its tests compare rounded metrics, not hashes.

## 7. Threats to validity

- Construct: actions stand in for responses and a 16 x 4 linear map
  stands in for a model. The constructs "fine-tuning", "preference"
  and "forgetting" are analogies at this scale.
- Internal: the label rule builds a rank-2 change from the base, so
  a rank-2 adapter is favoured by construction (stated in
  lora_learn.py). The DPO preferences are generated from the same
  checker that scores the task.
- External: no result here predicts behaviour on a language model.
  The shared-weight DPO run narrows the gap in kind (generalisation
  and spill-over now exist) but not in scale.
- Statistical: covered in section 5.
- Measurement: the QLoRA study counts weight bytes only; optimizer
  state, gradients, activations and quantisation scales are excluded,
  so it is a floor, not a memory measurement.

## 8. Related work

Before: NOVELTY.md is a draft with no checked citations, and its
claim gate refers to a check a reader cannot verify from the repo.

Fixed here, partly: the primary sources for each method the code
implements, so a reader can check the definitions against them.

- LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language
  Models", 2021, arXiv:2106.09685. Defines W = W0 + (alpha / r) B A
  with a frozen base, as in lora_learn.py.
- QLoRA: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized
  LLMs", 2023, arXiv:2305.14314. 4-bit base plus LoRA adapters, the
  setting memory_study.py accounts for.
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms",
  2017, arXiv:1707.06347. The clipped surrogate used in grpo_build.py.
- GRPO: Shao et al., "DeepSeekMath: Pushing the Limits of
  Mathematical Reasoning in Open Language Models", 2024,
  arXiv:2402.03300. Group-relative advantages replacing the critic.
- DPO: Rafailov et al., "Direct Preference Optimization: Your
  Language Model is Secretly a Reward Model", 2023, arXiv:2305.18290.
  The loss in dpo_deep.py.

Open: positioning against prior measured results (for example, how
reported LoRA-versus-full gaps on real models compare with this toy)
needs a literature pass with the papers' tables in hand, and only
makes sense once section 1's real-model run exists. The novelty
claim stays a draft.

## 9. Limitations

Already the repo's strongest section: every level README names its
deductions, and the Deep grade (8,000 of 10,000) is itemized. This
change carries those limits onto the dashboard, labels every live
panel as a JavaScript-stream run beside the recorded numbers, and
keeps the "accounting, not measured GPU memory" label on the QLoRA
calculator.

## Open gaps and what each needs

| Gap | Needs |
| --- | --- |
| Real open-model LoRA SFT, GRPO and DPO (ROADMAP 1) | One 0.5B to 1.5B open model, one GPU with 24 GB or more, a few GPU-hours per method (estimate), public instruction and math-checker datasets |
| Measured QLoRA memory (ROADMAP 2) | A 4-bit quantisation library, a GPU, and a 7B-class model download; measure peak allocated memory against the accounting table |
| PPO with a critic, DPO against GRPO at matched compute | Same model and GPU as row 1, roughly twice the GPU time for the critic |
| Human or model preference data (ROADMAP 6) | A public labelled preference dataset and the row 1 setup |
| Cross-project harness scoring (ROADMAP 7) | The row 1 runs, then the P06 evaluation harness |
| Safetensors checkpoints (ROADMAP 8, second half) | The safetensors package; a format change with no research effect, left for the real-model run |

Everything on the CPU side that the rubric asks for (seed variance,
baselines, ablations, sweeps, a stronger forgetting test, early
stopping, CI on the full suite, pinned dependencies) is done in this
change.
