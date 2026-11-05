# ROADMAP: finetune-lab (P10), v1.1 backlog

Status: v1.0 shipped 2026-11-05 (Learn, Build, and compressed
Deep). This file names the rest of the v2 Deep scope that the
December compression protocol moved out of v1.0. Nothing here
is done, and nothing here is claimed.

## v1.1 backlog (named v2 Deep scope, not in v1.0)

1. Real open-model run: DPO and GRPO on a small open language
   model with real text, a tokenizer, and sequence
   log-probabilities, scored before and after. This is the
   largest named gap in every level grade so far.
2. QLoRA measured study: load a 4-bit quantized model, train
   a LoRA adapter on it, and measure real GPU memory against
   the accounting table in level-3-deep, reporting where the
   accounting floor and the measured total differ (optimizer
   state, activations, quantization scales).
3. Stronger forgetting checks: a misaligned or multi-task
   preference set, plus an unrelated capability eval before
   and after DPO, so forgetting is tested where training and
   the retained task pull in different directions.
4. GRPO budget study in the P03 style: accuracy against
   training cost for the GRPO run, with a beta sweep for the
   KL term (Build named beta 0.04 letting KL drift to 1.1795
   nats; the sweep prices that trade).
5. Hyperparameter sweeps for DPO: beta, learning rate, and
   steps, with margin, held-out preference accuracy, sampled
   reward, and KL reported per setting, plus an implicit
   reward accuracy check on unseen prompts, not only unseen
   rejected contrasts.
6. Human or model preference data: replace or extend the
   synthetic checker-rule preferences with labelled pairs,
   and report how the results change.
7. P06 harness scoring: score the open-model runs with the
   rag-eval-harness from P06, closing the cross-project
   scoring gap named at Build.
8. Early stopping and best-checkpoint selection for the SFT
   pipeline (Build ships the final epoch adapter although
   validation peaks earlier), and safetensors checkpoints
   in place of npz.
9. Fan-out completion for v1.0: blog post, tutorial, website
   page copy, and LinkedIn draft graded on the publicity
   rubric, with outcomes logged in PUBLICITY.md. Prepared by
   the loop under the standing publicity gate; not part of
   the v1.0 code release.

## Order note

Items 1 and 2 need a GPU window (BlackBox first, per the
fleet rule) and a model download. Items 3 to 6 are CPU
teaching scale work that can start without one.
