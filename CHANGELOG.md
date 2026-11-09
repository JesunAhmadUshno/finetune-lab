# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning. The tags v0.1, v0.5 and v1.0 were
cut under the earlier Learn / Build / Deep ladder; under the portfolio
standard v2 the repo is Functional, not v1.0 Released.

## [Unreleased]

### Changed
- 2026-11-08: Restructured to the portfolio standard v2. README is the
  abstract (question, headline with its interval, verdict). SPEC.md and
  NOVELTY.md move into PROPOSAL.md; DOSSIER.md, ROADMAP.md,
  docs/research-review.md and the README benchmark table move into
  REPORT.md; the README fan-out section moves into PUBLICITY.md.
  Grades and rubric scores are removed from every public file
  (READMEs, fan-out drafts, PUBLICITY.md, the dashboard); the level
  README grade sections become point-free limit lists. The dashboard
  shows the Deep limits, the REPORT backlog and the README verdict.
- 2026-11-08: CI runs on Python 3.11 and 3.12 with read-only token
  permissions and checks that the results regenerate from the runners
  (scripts/check_results.py). CodeQL, Dependabot, CITATION.cff and
  SECURITY.md added. tests/test_smoke.py is now the layout test.
- 2026-11-07: vite upgraded to 6.4.5 (npm audit: no high or critical).

### Added
- 2026-11-07: Live dashboard (site/) and its Pages workflow.
- 2026-11-07: Rigor study (rigor/): 10-seed intervals, SFT baselines
  and early stopping, GRPO beta sweep and ablations, a 36-setting DPO
  sweep, misaligned and shared-weight retention checks. Found and
  documented the Build GRPO gradient normalization defect (decision
  0001). Research review written (now in REPORT.md).
- 2026-11-07: CI runs the full unittest suite; numpy pinned to 1.26.4.

## [1.0] - 2026-11-05

### Added
- Deep: DPO run with forgetting checks, a reward hacking probe and a
  QLoRA memory accounting study. Written fan-out set in fanout/.

## [0.5] - 2026-11-05

### Added
- Build: SFT LoRA pipeline (data prep, minibatch training, checkpoint
  with hash verification, eval harness) and a GRPO track.

## [0.1] - 2026-11-05

### Added
- Learn: LoRA from scratch and hand-checked GRPO advantages.
- 2026-11-04: Scaffold: SPEC.md, DOSSIER.md, NOVELTY.md and the
  Learn / Build / Deep level folders.
