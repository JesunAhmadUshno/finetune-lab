# 0001: Keep the recorded GRPO results despite the gradient defect

Date: 2026-11-07. Status: accepted.

Context: the rigor study found that level-2-build/grpo_build.py
normalizes prompt p's surrogate gradient by (p + 1) x 8 instead of 64
and does not average the KL gradient over prompts, so the gradient does
not match the stated loss (tests/test_rigor.py shows it by finite
differences).

Decision: keep the Build and Deep results as run, and add a corrected
gradient (rigor_study.grpo_grad_corrected) as an ablation beside them.

Consequences: the history stays honest and checkable. Readers must use
the ablation row (10-seed mean reward 0.9180, CI 0.9027 to 0.9332)
when they want the stated-loss behaviour. The dashboard offers the
corrected rule as a toggle.
