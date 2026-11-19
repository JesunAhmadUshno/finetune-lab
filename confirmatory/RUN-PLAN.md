# Confirmatory run plan (implementation choices inside the 0006 lock)

Date: 2026-11-17, written before the first confirmatory run.
The lock is docs/decisions/0006-confirmatory-lock-values.md. The
lock fixes the hypotheses, pass lines, seed count, verdict rule,
epochs for the H1 arms, the adapter learning rate, and the masking
convention. The choices below are implementation settings the lock
does not name; they are fixed here, before any run, and reported
with the results. Changing any of them after the first run starts
needs a dated amendment.

## Fixed choices

- Seeds: 0 to 9 for every arm. No seed dropped; every run
  reported.
- SFT data: Alpaca shuffled with seed 0. Train slice: shuffled
  rows 0 to 9,999. Validation slice (checkpoint selection): the
  next 1,000. Held-out slice: the next 1,000 (recorded for the
  record; H1's held-out accuracy is the GSM8K test split, per the
  lock). Examples whose tokenized length exceeds 512 are dropped
  and counted.
- SFT prompt format (Alpaca template, the dataset's own fields):
  "### Instruction:\n{instruction}" plus, when input is non-empty,
  "\n\n### Input:\n{input}", then "\n\n### Response:\n{output}"
  with the EOS token appended to the completion. Loss on
  completion tokens only (the locked masking convention).
- Full fine-tune arm: learning rate 1e-5 (the standard full-model
  rate the lock leaves to the arm), AdamW, cosine schedule,
  warmup 3 percent, effective batch 64 (16 per step, 4 gradient
  accumulation steps), bf16 autocast over fp32 weights, 3 epochs
  (locked), best-validation checkpoint selected at epoch ends on
  validation completion loss.
- LoRA arm: learning rate 1e-3 (locked), LoRA rank 16, alpha 32,
  dropout 0, target modules q, k, v, o, gate, up, down projections
  (the all-linear convention of the TRL-class recipes the repo
  names as prior art); same data, epochs, batch, schedule, and
  selection rule as the full arm (matched budget). The selected
  adapter is merged for evaluation.
- GSM8K evaluation protocol (one protocol for the base model and
  every tuned arm; used for H1, and for H4's pre and post
  accuracies): zero-shot prompt "### Instruction:\nSolve the
  following math word problem. Give the final numeric answer
  after '####'.\n\n### Input:\n{question}\n\n### Response:\n";
  greedy decoding, at most 320 new tokens. Extraction: the number
  after the last "####" when present, else the last number in the
  completion; commas, dollar signs, percent signs, and spaces
  stripped; numeric equality with the gold final answer.
- DPO starting point: each seed's DPO arms (aligned and
  misaligned) start from that same seed's full fine-tune SFT
  checkpoint (best-validation). The H3 reference is that same
  checkpoint. The H4 pre-DPO accuracy is that checkpoint's GSM8K
  test accuracy from the H1 evaluation.
- DPO pairs: prompts in GSM8K train order, the first 2,000.
  Chosen: the gold solution with calculator annotations stripped.
  Rejected: one completion sampled from the seed's SFT policy at
  temperature 0.9 (up to 4 samples, then up to 2 resample rounds)
  whose extracted answer fails the checker; a prompt with no
  failing sample is skipped and counted. The misaligned arm flips
  chosen and rejected on 50 percent of prompts: the first 1,000
  pairs in pair order are flipped (the deterministic mirror of the
  post hoc design's 4 of 8).
- Held-out preference pairs (H3 measurement): all 1,319 GSM8K
  test prompts, same construction, rejected sampled from the
  seed's SFT reference policy. Preference accuracy of a policy:
  the fraction of pairs where the policy's summed completion
  log-probability of chosen exceeds that of rejected. Pairs where
  no failing sample exists are excluded from the denominator and
  counted.
- DPO training: full-model, from the seed's SFT checkpoint,
  reference frozen at that checkpoint, beta 0.1, learning rate
  5e-7, AdamW, cosine, 1 epoch over the seed's pairs, effective
  batch 16 pairs.
- GRPO: start policy and frozen reference are the seed's full
  fine-tune SFT checkpoint. Prompts: GSM8K train shuffled with
  the run seed; the first 256 are training prompts, the next 64
  are the frozen final-measurement prompts. Group size 8,
  temperature 1.0, at most 320 new tokens, reward 1 when the
  extracted answer equals gold, else 0. Advantage: group
  normalized, (r minus group mean) over (group std plus 1e-4).
  Objective: clipped policy gradient (clip 0.2) minus beta times
  the k3 KL estimator against the reference, one on-policy pass
  per generated minibatch, minibatch 32 prompt groups, learning
  rate 1e-6, 2 passes over the 256 prompts. Final KL and final
  reward: measured after training on the 64 frozen measurement
  prompts, 8 samples each at temperature 1.0; final reward is the
  mean reward, final KL is the mean k3 estimate.
- GPU-hour accounting: the driver logs wall-clock start and end
  for every unit (one seed of one arm, one evaluation, one pair
  build). The sum is the GPU-hour figure reported against the
  96-hour cap.
- Arm order: H1, then H3 and H4 (they share the pair builds),
  then H2. The PPO baseline of section 4 runs last, and only if
  the hours used by H1 to H4 leave comfortable room under the
  cap; if it does not run, the results record says so with the
  hours figure.
- Checkpointing: every unit writes its metrics and raw outputs
  when it finishes; the driver skips units whose completion
  marker exists, so an interruption loses at most the unit in
  flight.

## Pre-run implementation note (2026-11-17, before the first confirmatory run)

A smoke pass over shrunken slices (seed 0 only; 200 training
examples, 100 evaluation items, 64 DPO pairs, 16 GRPO prompts)
validated every code path and measured unit times. The measured
times projected well past the budget at the original step sizes,
so the following step-size choices were set before the first
confirmatory run; none of them touches a locked value (the
effective batch stays 64, the data, epochs, rates, and selection
rules are unchanged):

- SFT: per-device batch 32 with 2 gradient accumulation steps,
  fused AdamW, no gradient checkpointing (the 1.5B model at
  sequence length 512 fits the machine's memory without it).
- DPO and GRPO training forwards: no gradient checkpointing.
- GRPO sampling: a bf16 copy of the policy, re-synced from the
  fp32 master weights before every sampling round, produces the
  samples and their old log-probabilities; training and the KL
  reference stay in fp32 and bf16 respectively, as before.
- The driver waits for at least 30 GiB of available memory before
  starting a unit and retries a failed unit once (the shared
  machine showed one transient out-of-memory during the smoke
  pass, in a unit that then ran cleanly).
