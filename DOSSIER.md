# DOSSIER: finetune-lab (P10)

Updated to ship state at v1.0, 2026-11-05. First version
2026-11-04 (v1). Written under the documentation and
publishing standard (past, present, future).

## Past

Prior art for this project is named in NOVELTY.md (fine-tuning
recipes in framework docs (TRL class)). The history of the
problem, the key papers and systems, and what was tried before
are recorded in the Roadmap v2 section text in SPEC.md. The
methods used here are the standard published ones, rebuilt
from scratch in NumPy for teaching: LoRA (low-rank adapter
factors on a frozen base), GRPO (group-relative advantages
with a clipped surrogate and a KL term), DPO (direct
preference optimisation with a frozen reference policy and a
beta-scaled log-sigmoid loss), and QLoRA (a 4-bit quantized
base with a low-rank adapter, studied here as accounting).

## Present

Current demand evidence: hiring value 5 of 5 (post-training
duties at Cohere, Mistral, OpenAI; v2 rank 5) [confirmed,
fleet and scoring pass 2026-11-04; for P23 to P29, the
decision pack 2026-11-04]. The Round 3 research (2026-11-04)
found the market centre at post-training, evals, environments,
and inference depth [confirmed as the research finding; the
report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md].

Shipped state, v1.0, 2026-11-05, all on synthetic NumPy tasks
on Muse VM CPU, no real LLM and no real text data at any
level:

- Learn (v0.1): LoRA from scratch and GRPO advantages,
  hand-checked. Test accuracy 0.5469 to 0.9219 on its slice.
- Build (v0.5): a full SFT-style LoRA pipeline (data prep,
  minibatch training, checkpoint with hash verification, eval
  harness) raising held-out test accuracy from 0.6797 to
  0.9141, plus a GRPO track training a tabular toy policy
  from greedy accuracy 0.125 (1 of 8) to 1.0000 (8 of 8) and
  fresh-sample mean reward 0.2070 to 0.9766. Grade 8,300 of
  10,000, deductions in level-2-build/README.md.
- Deep (v1.0, compressed Deep under the December protocol):
  a DPO run on the Build tabular policy (beta 0.1, frozen
  reference, 300 full-batch steps, hand-derived gradient)
  raising held-out preference margin from -0.0654 to 0.8297
  and held-out preference accuracy from 0.4375 (7 of 16) to
  1.0000 (16 of 16); forgetting checks on the Build GRPO
  task showing no forgetting on that aligned task (greedy
  change +0.875, mean reward change +0.336, mean KL 0.1833,
  untouched-action retention 1.0000, with the weakness of
  the test named in the level README); a reward hacking
  probe showing a rejected-only suppression policy games the
  train contrast to 1.0000 while held-out accuracy stays
  0.4375; and a QLoRA memory accounting study (not measured
  GPU memory) putting a stated 7B-scale model's 4-bit base
  plus rank 16 adapter at 3,516,777,216 bytes against
  14,000,000,000 fp16 full bytes. Grade 8,000 of 10,000,
  deductions in level-3-deep/README.md.
- Tests: 39 in the repo suite, all passing, including 11 new
  Deep tests.

## Future

The rest of the v2 Deep scope is the v1.1 backlog in
ROADMAP.md: a real open-model run, a measured QLoRA study,
stronger (misaligned) forgetting checks, the GRPO budget and
KL beta study, DPO hyperparameter sweeps, labelled preference
data, and P06 harness scoring. Items needing a GPU window
and a model download are named as such there. Any forecast
about this field is [unverified] unless a dated primary
source is added here.
