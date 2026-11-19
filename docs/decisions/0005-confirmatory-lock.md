# 0005: Confirmatory re-run lock: PROPOSAL.md section 4 names no defaults, so every bracket is recorded still open

Date: 2026-11-17. Status: accepted.
Pointer (added 2026-11-17): the brackets left open by this entry
were settled by 0006-confirmatory-lock-values.md, which records
the locked design. This entry's content is unchanged.

Context: on 2026-11-17 the Owner ruled that the confirmatory
re-run drafted in PROPOSAL.md section 4 locks with that section's
own defaults. Section 4 was therefore re-read in full to extract
them. It names none. Its own words govern: "The settings in
brackets are decisions for the Owner, not facts", and its pass
lines stand as "[to be set before the lock]". Section 4 as written
(quoted verbatim) is:

- "Model: one open model of roughly 0.5B to 1.5B parameters [which
  model]."
- "Compute: one GPU with at least 24 GB [which machine, how many
  GPU-hours]."
- "Data: a public instruction set and a public math word-problem
  set with a programmatic checker [which sets]."
- "H1: a LoRA adapter reaches held-out accuracy within [x] points
  of full fine-tuning at matched budget."
- "H2: the GRPO KL term trades reward for closeness to the
  reference: higher beta gives lower final KL and lower reward."
- "H3: DPO on aligned preferences improves held-out preference
  accuracy over the reference by at least [x]."
- "H4: DPO on misaligned preferences lowers accuracy on an
  unrelated retained task by at least [x]."
- "Analysis: [n] seeds per arm, bootstrap 95% intervals, every run
  reported, baselines (full fine-tune, PPO with a critic) under the
  same budget."
- "Pass lines and the CONFIRMED / REFUTED / INCONCLUSIVE rule:
  [to be set before the lock]."

Decision: the lock is recorded exactly as the proposal stands.
No default is invented for any bracket, and no bracket is settled
by this entry. The state of every bracket is:

Still open, the proposal's own brackets:
- Which model (the 0.5B to 1.5B range is stated; no model is
  named).
- Which machine and how many GPU-hours (at least 24 GB is stated;
  no machine or budget is named).
- Which instruction set and which math word-problem set.
- The H1 pass line, [x] points.
- The H3 pass line, at least [x].
- The H4 pass line, at least [x].
- The seed count, [n] per arm.
- The pass lines as a set, and the CONFIRMED / REFUTED /
  INCONCLUSIVE rule that applies them.

Still open, the brackets named in the Owner's 2026-11-17 ruling
brief, none of which appears in PROPOSAL.md or anywhere else in
this repo's record, so no default exists to lock:
- Training epochs for the proposal, 2 vs 3.
- Adapter learning rate, 1e-3 vs 5e-4.
- The masking convention for the H1 arms (two conventions).

Consequences: the confirmatory design is NOT locked in any part
that carries a bracket, and the re-run does not start under this
entry alone. It also waits on a GPU window with at least 24 GB,
which no current fleet window provides. What would close the
lock: the Owner naming a value for each bracket above (or
amending PROPOSAL.md section 4 to name defaults), after which a
further entry records the settled design and the proposal-v1 tag
can be applied. PROPOSAL.md itself is not edited by this entry;
its header still reads "Status: NOT LOCKED", which remains
accurate.
