# 0003: Not pre-registered; no lock after the fact

Date: 2026-11-08. Status: accepted.

Context: SPEC.md was committed in the first commit (164712d,
2026-11-04), before any code. It sets scope and acceptance criteria but
no hypotheses, pass lines or analysis plan. The research review of
2026-11-07 states that no hypothesis was written before any run.

Decision: the repo is labelled "not pre-registered" and capped at
Functional. No proposal-v1 tag is placed on 164712d or on any later
commit, because a lock placed after results exist proves nothing.

Consequences: the next step is the confirmatory re-run drafted in
PROPOSAL.md section 4. Its settings and pass lines are the Owner's to
set; it is locked (tag proposal-v1) before any of its runs.
